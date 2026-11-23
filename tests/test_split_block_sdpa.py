import copy
import os
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import Block, SelfAttention, precompute_rope_cos_sin  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402


def _make_block_diag_masks(levels, prefix_len, pad_mask=None):
    target_len = sum(levels)
    d = torch.cat([torch.full((lvl,), i) for i, lvl in enumerate(levels)]).view(1, target_len, 1)
    d_t = d.transpose(1, 2)
    target_block = torch.where(d == d_t, 0.0, float("-inf")).reshape(target_len, target_len)

    full_mask = torch.cat(
        [
            torch.cat([torch.zeros(prefix_len, prefix_len), torch.full((prefix_len, target_len), float("-inf"))], dim=1),
            torch.cat([torch.zeros(target_len, prefix_len), target_block], dim=1),
        ],
        dim=0,
    ).reshape(1, 1, prefix_len + target_len, prefix_len + target_len)

    target_keep = (~torch.isneginf(target_block)).reshape(1, 1, target_len, target_len)
    if pad_mask is None:
        prefix_attn_mask = None
        target_attn_mask = torch.cat(
            [torch.ones((1, 1, target_len, prefix_len), dtype=torch.bool), target_keep],
            dim=-1,
        )
    else:
        batch_size = pad_mask.shape[0]
        pad_bias = torch.zeros((batch_size, 1, 1, prefix_len + target_len), dtype=torch.float32)
        pad_bias[:, :, :, :prefix_len] = torch.where(
            pad_mask[:, None, None, :],
            torch.tensor(float("-inf")),
            torch.tensor(0.0),
        )
        full_mask = full_mask + pad_bias
        prefix_attn_mask = ~pad_mask[:, None, None, :]
        target_attn_mask = torch.cat(
            [
                prefix_attn_mask.expand(batch_size, 1, target_len, prefix_len),
                target_keep.expand(batch_size, 1, target_len, target_len),
            ],
            dim=-1,
        )
    return full_mask, prefix_attn_mask, target_attn_mask


def _make_attn_cfg(**overrides):
    defaults = dict(
        n_embd=32,
        n_head=4,
        n_kv_heads=4,
        bias=True,
        dropout=0.0,
        use_qk_norm=True,
        parallel_block=False,
        normformer=False,
        use_layerscale=False,
        use_rezero=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build_ncm_pair(
    *,
    levels=(2, 4, 8),
    prefix_len=12,
    query_level0=False,
    predict_level0=False,
    l0_autoregressive=False,
    use_rope=True,
    attn_pattern="block_diagonal",
):
    vqvae = VQVAE(
        in_vocab_size=32,
        context_length=max(prefix_len, levels[-1]),
        embed_dim=16,
        vqvae_vocab_size=[16] * len(levels),
        n_layers=2,
        compression_factor=1,
        dropout=0.0,
        enc_layers=0,
        dec_layers=1,
        vector_quantizer_config=OmegaConf.create(
            {
                "clss": "multiscale_residual_vector_quantizer",
                "decay": 0.99,
                "commitment_cost": 0.25,
                "levels": {
                    "use_manual_levels": True,
                    "manual_levels": list(levels),
                },
            }
        ),
    )
    target_levels = levels if query_level0 else levels[1:]
    cfg = OmegaConf.create(
        {
            "n_layer": 2,
            "n_head": 4,
            "n_kv_heads": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "prefix_len": prefix_len,
            "vocab_size": [16] * len(target_levels),
            "attn_pattern": attn_pattern,
            "use_level_encoding": True,
            "use_rope": use_rope,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": query_level0,
            "l0_autoregressive": l0_autoregressive,
            "predict_level0": predict_level0,
            "prefix_pad_token_id": 0,
            "tie_heads_to_codebook": False,
        }
    )
    model_split = NextConceptTransformer(vqvae, cfg)
    model_ref = copy.deepcopy(model_split)
    model_split.train()
    model_ref.train()
    return model_split, model_ref


def _full_mask_ncm_forward(model, x, prefix_pad_mask=None):
    z_prefix = x[:, :model.prefix_len, :]
    x_target = x[:, model.prefix_len:, :]
    prefix_emb = model.transformer.wte(z_prefix)
    tgt_emb = model.transformer.wte(x_target)
    if not model.use_rope:
        prefix_emb = prefix_emb + model.wpe(model.wp_prefix)
        tgt_emb = tgt_emb + model.wpe(model.wp_1L)
    if model.config.use_level_encoding:
        tgt_emb = tgt_emb + model.transformer.wle(model.lvl_1L)

    h = torch.cat([prefix_emb, tgt_emb], dim=1)
    attn_mask = model.attn_mask_for_masking
    if prefix_pad_mask is not None:
        attn_mask = attn_mask + model._prefix_pad_bias(
            prefix_pad_mask, h.shape[1], attn_mask.dtype, h.device
        )
    rope = (model.rope_cos, model.rope_sin) if model.use_rope else None
    for block in model.transformer.blocks:
        h = block(h, attn_mask=attn_mask, rope_cos_sin=rope)
    h = model.transformer.ln_f(h)

    h_target = h[:, model.prefix_len:, :]
    logits_list = []
    if model.predict_level0:
        l0_h = model._level0_slice(h, prefix_pad_mask)
        logits_list.append(model._pad_vocab(model.lm_head_l0(l0_h)))
    offset = 0
    for i, level in enumerate(model.levels[model._lvl_first:]):
        logits_list.append(model._pad_vocab(model.lm_heads[i](h_target[:, offset : offset + level, :])))
        offset += level
    return torch.cat(logits_list, dim=1)


def _assert_grads_close(mod_a, mod_b, atol=1e-5, rtol=1e-5):
    params_a = dict(mod_a.named_parameters())
    params_b = dict(mod_b.named_parameters())
    assert set(params_a) == set(params_b)
    for name, pa in params_a.items():
        pb = params_b[name]
        if pa.grad is None and pb.grad is None:
            continue
        assert pa.grad is not None and pb.grad is not None, f"grad presence mismatch for {name}"
        assert torch.allclose(pa.grad, pb.grad, atol=atol, rtol=rtol), (
            f"grad mismatch for {name}: max diff={(pa.grad - pb.grad).abs().max().item()}"
        )


@pytest.mark.parametrize("use_pad", [False, True])
@pytest.mark.parametrize("n_kv_heads", [4, 2])
@pytest.mark.parametrize("module_kind", ["attn", "block", "parallel_block"])
def test_self_attention_and_block_split_sdpa_equivalence(module_kind, n_kv_heads, use_pad):
    torch.manual_seed(42)
    batch_size, prefix_len, levels = 2, 12, (2, 4, 8)
    total_len = prefix_len + sum(levels)
    cfg = _make_attn_cfg(n_kv_heads=n_kv_heads, parallel_block=(module_kind == "parallel_block"))

    cls = SelfAttention if module_kind == "attn" else Block
    mod_full = cls(cfg).train()
    mod_split = copy.deepcopy(mod_full).train()

    x_base = torch.randn(batch_size, total_len, cfg.n_embd)
    x_full = x_base.clone().requires_grad_(True)
    x_split = x_base.clone().requires_grad_(True)

    pad_mask = None
    if use_pad:
        pad_mask = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        pad_mask[0, -4:] = True
        pad_mask[1, -7:] = True

    full_mask, prefix_attn_mask, target_attn_mask = _make_block_diag_masks(
        levels, prefix_len, pad_mask=pad_mask
    )
    pos = torch.cat([torch.arange(prefix_len), *[torch.arange(l) for l in levels]])
    rope = precompute_rope_cos_sin(pos, cfg.n_embd // cfg.n_head)

    y_full = mod_full(x_full, attn_mask=full_mask, rope_cos_sin=rope)
    y_split = mod_split(
        x_split,
        attn_mask=target_attn_mask,
        rope_cos_sin=rope,
        split_prefix_len=prefix_len,
        prefix_attn_mask=prefix_attn_mask,
    )

    assert y_split.shape == y_full.shape
    assert torch.allclose(y_split, y_full, atol=1e-5, rtol=1e-5)

    weight = torch.randn_like(y_full)
    (y_full * weight).sum().backward()
    (y_split * weight).sum().backward()

    assert torch.allclose(x_split.grad, x_full.grad, atol=1e-5, rtol=1e-5)
    _assert_grads_close(mod_split, mod_full, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_pad", [False, True])
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize(
    "predict_level0,query_level0,l0_autoregressive",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_next_concept_transformer_split_sdpa_equivalence(
    predict_level0, query_level0, l0_autoregressive, use_rope, use_pad
):
    torch.manual_seed(123)
    levels = (2, 4, 8)
    prefix_len = 12
    model_split, model_ref = _build_ncm_pair(
        levels=levels,
        prefix_len=prefix_len,
        query_level0=query_level0,
        predict_level0=predict_level0,
        l0_autoregressive=l0_autoregressive,
        use_rope=use_rope,
        attn_pattern="block_diagonal",
    )

    target_len = sum(levels) if query_level0 else sum(levels[1:])
    batch_size = 2
    x_base = torch.randn(batch_size, prefix_len + target_len, 16)
    x_split = x_base.clone().requires_grad_(True)
    x_ref = x_base.clone().requires_grad_(True)

    pad_mask = None
    if use_pad:
        pad_mask = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        pad_mask[0, -3:] = True
        pad_mask[1, -6:] = True

    logits_split = model_split(x_split, prefix_pad_mask=pad_mask)
    logits_ref = _full_mask_ncm_forward(model_ref, x_ref, prefix_pad_mask=pad_mask)

    assert logits_split.shape == logits_ref.shape
    assert torch.allclose(logits_split, logits_ref, atol=1e-5, rtol=1e-5)

    targets = torch.randint(0, 16, (batch_size, logits_split.shape[1]))
    loss_split = torch.nn.functional.cross_entropy(
        logits_split.reshape(-1, logits_split.shape[-1]), targets.reshape(-1)
    )
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.reshape(-1, logits_ref.shape[-1]), targets.reshape(-1)
    )
    loss_split.backward()
    loss_ref.backward()

    assert torch.allclose(x_split.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    _assert_grads_close(model_split, model_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("predict_level0", [False, True])
def test_split_sdpa_all_pad_prefix_row_stays_finite(predict_level0):
    torch.manual_seed(7)
    levels = (2, 4, 8)
    prefix_len = 12
    model_split, _ = _build_ncm_pair(
        levels=levels,
        prefix_len=prefix_len,
        predict_level0=predict_level0,
        query_level0=False,
        attn_pattern="block_diagonal",
    )

    target_len = sum(levels[1:])
    x = torch.randn(2, prefix_len + target_len, 16, requires_grad=True)
    pad_mask = torch.tensor(
        [
            [True] * prefix_len,
            [False, False, False] + [True] * (prefix_len - 3),
        ],
        dtype=torch.bool,
    )

    logits = model_split(x, prefix_pad_mask=pad_mask)
    assert torch.isfinite(logits).all()

    logits.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, param in model_split.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"


def test_block_causal_and_kv_cached_generate_untouched():
    torch.manual_seed(99)
    levels = (2, 4, 8)
    prefix_len = 12

    model_causal, model_causal_ref = _build_ncm_pair(
        levels=levels,
        prefix_len=prefix_len,
        attn_pattern="block_causal",
    )
    target_len = sum(levels[1:])
    x = torch.randn(2, prefix_len + target_len, 16)
    out_causal = model_causal(x)
    out_causal_ref = _full_mask_ncm_forward(model_causal_ref, x)
    assert torch.allclose(out_causal, out_causal_ref, atol=1e-6, rtol=1e-6)

    model_gen, _ = _build_ncm_pair(
        levels=levels,
        prefix_len=prefix_len,
        predict_level0=True,
        attn_pattern="block_diagonal",
    )
    model_gen.eval()
    prefix_tokens = torch.randint(1, 32, (2, prefix_len))
    prefix_tokens[0, -4:] = 0
    gen_cached = model_gen.generate(
        None, prefix_tokens, temperature=0.0, use_kv_cache=True
    )
    gen_uncached = model_gen.generate(
        None, prefix_tokens, temperature=0.0, use_kv_cache=False
    )
    assert gen_cached.shape == (2, sum(levels))
    assert torch.equal(gen_cached, gen_uncached)


def test_split_sdpa_bf16_autocast_and_state_dict_compatibility():
    torch.manual_seed(11)
    levels = (2, 4, 8)
    prefix_len = 12
    model, _ = _build_ncm_pair(levels=levels, prefix_len=prefix_len, attn_pattern="block_diagonal")

    sd_keys = set(model.state_dict().keys())
    for buf_name in ("target_keep_mask_bool", "no_pad_target_mask_bool", "prefix_col0"):
        assert buf_name not in sd_keys

    x = torch.randn(2, prefix_len + sum(levels[1:]), 16)
    pad_mask = torch.zeros(2, prefix_len, dtype=torch.bool)
    pad_mask[0, -3:] = True

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x, prefix_pad_mask=pad_mask)
        loss = logits.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(logits).all()


def test_split_sdpa_cpu_eager_smoke_benchmark():
    torch.manual_seed(0)
    levels = (2, 4, 8, 16, 32)
    prefix_len = 128
    model, _ = _build_ncm_pair(
        levels=levels,
        prefix_len=prefix_len,
        attn_pattern="block_diagonal",
    )
    model.eval()
    x = torch.randn(2, prefix_len + sum(levels[1:]), 16)

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert logits.shape == (2, sum(levels[1:]), 16)
    assert torch.isfinite(logits).all()
    assert elapsed_ms >= 0.0

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import SelfAttention  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402


def _make_model(
    *,
    use_rope: bool,
    query_level0: bool,
    predict_level0: bool = False,
    l0_autoregressive: bool = False,
    use_qk_norm: bool = False,
    use_ctx_encoder: bool = False,
    prefix_len: int = 16,
    levels: tuple[int, ...] = (2, 4, 8),
    attn_pattern: str = "block_causal",
    n_kv_heads: int = 2,
) -> NextConceptTransformer:
    torch.manual_seed(42)
    embed_dim = 8
    n_head = 2
    n_embd = 16
    vocab_sizes_vq = [32 for _ in levels]
    head_vocabs = vocab_sizes_vq if query_level0 else vocab_sizes_vq[1:]

    vqvae_stub = SimpleNamespace(
        embed_dim=embed_dim,
        in_vocab_size=64,
        vq=SimpleNamespace(
            levels=list(levels),
            vqvae_vocab_sizes=vocab_sizes_vq,
        ),
    )
    cfg = SimpleNamespace(
        n_embd=n_embd,
        n_layer=2,
        n_head=n_head,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
        bias=False,
        vocab_size=head_vocabs,
        attn_pattern=attn_pattern,
        query_level0=query_level0,
        l0_autoregressive=l0_autoregressive,
        prefix_len=prefix_len,
        use_rope=use_rope,
        rope_base=10000,
        use_level_encoding=True,
        use_flex_attention=False,
        use_qk_norm=use_qk_norm,
        predict_level0=predict_level0,
        ctx_encoder=SimpleNamespace(enabled=use_ctx_encoder, n_layer=1, vocab_size=64),
    )
    model = NextConceptTransformer(vqvae_stub, cfg)
    model.eval()
    return model


def _build_real_vqvae_model(
    *,
    levels: tuple[int, ...] = (2, 4, 8),
    prefix_len: int = 12,
    query_level0: bool = False,
    predict_level0: bool = True,
    use_rope: bool = True,
    use_qk_norm: bool = True,
    attn_pattern: str = "block_diagonal",
) -> NextConceptTransformer:
    torch.manual_seed(7)
    levels_list = list(levels)
    vqvae = VQVAE(
        in_vocab_size=32,
        context_length=max(prefix_len, sum(levels_list)),
        embed_dim=16,
        vqvae_vocab_size=[16] * len(levels_list),
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
                    "manual_levels": levels_list,
                },
            }
        ),
    ).eval()
    target_levels = levels_list if query_level0 else levels_list[1:]
    cfg = OmegaConf.create(
        {
            "n_layer": 2,
            "n_head": 4,
            "n_kv_heads": 2,
            "bias": False,
            "dropout": 0.0,
            "n_embd": 32,
            "prefix_len": prefix_len,
            "prefix_pad_token_id": 0,
            "vocab_size": [16] * len(target_levels),
            "attn_pattern": attn_pattern,
            "use_level_encoding": True,
            "use_rope": use_rope,
            "rope_base": 10000,
            "use_qk_norm": use_qk_norm,
            "use_flex_attention": False,
            "query_level0": query_level0,
            "l0_autoregressive": False,
            "predict_level0": predict_level0,
            "tie_heads_to_codebook": False,
        }
    )
    return NextConceptTransformer(vqvae, cfg).eval()


def _build_inputs(
    model: NextConceptTransformer,
    batch_size: int,
    valid_lens: list[int],
    p_eff: int,
    *,
    requires_grad: bool = False,
):
    p_full = model.prefix_len
    t_target = sum(model.levels[model._lvl_first :])
    d_in = model.vqvae_local[0].embed_dim

    x_base = torch.randn(batch_size, p_full + t_target, d_in)
    pad_mask_full = torch.zeros(batch_size, p_full, dtype=torch.bool)
    for b, vlen in enumerate(valid_lens):
        if vlen < p_full:
            pad_mask_full[b, vlen:] = True

    x_full = x_base.clone().requires_grad_(requires_grad)
    x_trimmed = torch.cat([x_base[:, :p_eff, :], x_base[:, p_full:, :]], dim=1)
    if requires_grad:
        x_trimmed = x_trimmed.detach().requires_grad_(True)
    pad_mask_trimmed = pad_mask_full[:, :p_eff]
    return x_full, pad_mask_full, x_trimmed, pad_mask_trimmed


def _collect_grads(model: NextConceptTransformer) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("attn_pattern", ["block_diagonal", "block_causal"])
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("query_level0", [False, True])
def test_right_pad_trimming_exact_logit_and_grad_parity(
    attn_pattern: str, use_rope: bool, query_level0: bool
):
    model = _make_model(
        use_rope=use_rope,
        query_level0=query_level0,
        prefix_len=16,
        attn_pattern=attn_pattern,
    )
    valid_lens = [10, 6]
    p_eff = 10
    p_full = model.prefix_len
    x_full, pad_full, x_trim, pad_trim = _build_inputs(
        model, batch_size=2, valid_lens=valid_lens, p_eff=p_eff, requires_grad=True
    )

    logits_full = model(x_full, prefix_pad_mask=pad_full)
    assert torch.isfinite(logits_full).all()

    targets = torch.randint(0, 32, (2, logits_full.shape[1]))
    model.zero_grad(set_to_none=True)
    loss_full = F.cross_entropy(logits_full.reshape(-1, logits_full.shape[-1]), targets.reshape(-1))
    loss_full.backward()
    grads_full = _collect_grads(model)

    model.zero_grad(set_to_none=True)
    logits_trim_short_mask = model(x_trim, prefix_pad_mask=pad_trim)
    assert torch.allclose(logits_trim_short_mask, logits_full, atol=1e-6, rtol=1e-5)

    loss_trim = F.cross_entropy(
        logits_trim_short_mask.reshape(-1, logits_trim_short_mask.shape[-1]), targets.reshape(-1)
    )
    loss_trim.backward()
    grads_trim = _collect_grads(model)

    with torch.no_grad():
        logits_trim_full_mask = model(x_trim.detach(), prefix_pad_mask=pad_full)
    assert torch.equal(logits_trim_full_mask, logits_trim_short_mask.detach())

    assert torch.allclose(x_trim.grad[:, :p_eff], x_full.grad[:, :p_eff], atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_trim.grad[:, p_eff:], x_full.grad[:, p_full:], atol=1e-6, rtol=1e-5)
    assert torch.equal(x_full.grad[:, p_eff:p_full], torch.zeros_like(x_full.grad[:, p_eff:p_full]))
    assert set(grads_trim) == set(grads_full)
    for name in grads_full:
        assert torch.allclose(grads_trim[name], grads_full[name], atol=1e-6, rtol=1e-5), name


@pytest.mark.parametrize("use_rope", [False, True])
def test_right_pad_trimming_with_predict_level0_logit_and_grad_parity(use_rope: bool):
    model = _make_model(
        use_rope=use_rope,
        query_level0=False,
        predict_level0=True,
        prefix_len=16,
    )
    valid_lens = [11, 1]
    p_eff = 11
    p_full = model.prefix_len
    x_full, pad_full, x_trim, pad_trim = _build_inputs(
        model, batch_size=2, valid_lens=valid_lens, p_eff=p_eff, requires_grad=True
    )

    logits_full = model(x_full, prefix_pad_mask=pad_full)
    targets = torch.randint(0, 32, (2, logits_full.shape[1]))

    model.zero_grad(set_to_none=True)
    F.cross_entropy(logits_full.reshape(-1, logits_full.shape[-1]), targets.reshape(-1)).backward()
    grads_full = _collect_grads(model)

    model.zero_grad(set_to_none=True)
    logits_trim = model(x_trim, prefix_pad_mask=pad_trim)
    assert torch.isfinite(logits_full).all()
    assert torch.allclose(logits_trim, logits_full, atol=1e-6, rtol=1e-5)

    F.cross_entropy(logits_trim.reshape(-1, logits_trim.shape[-1]), targets.reshape(-1)).backward()
    grads_trim = _collect_grads(model)

    assert torch.allclose(x_trim.grad[:, :p_eff], x_full.grad[:, :p_eff], atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_trim.grad[:, p_eff:], x_full.grad[:, p_full:], atol=1e-6, rtol=1e-5)
    for name in grads_full:
        assert torch.allclose(grads_trim[name], grads_full[name], atol=1e-6, rtol=1e-5), name


def test_partial_padding_ignores_trailing_padded_values():
    model = _make_model(use_rope=True, query_level0=False, prefix_len=16)
    p_eff = 10
    x_full, pad_full, x_trim, pad_trim = _build_inputs(
        model, batch_size=2, valid_lens=[10, 4], p_eff=p_eff
    )

    x_trim_perturbed = x_trim.clone()
    x_trim_perturbed[1, 4:p_eff, :] += 50.0

    with torch.no_grad():
        logits_a = model(x_trim, prefix_pad_mask=pad_trim)
        logits_b = model(x_trim_perturbed, prefix_pad_mask=pad_trim)
        logits_full = model(x_full, prefix_pad_mask=pad_full)

    assert torch.allclose(logits_a, logits_full, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("predict_level0,query_level0", [(False, False), (True, False), (False, True)])
def test_all_pad_row_unmasks_col0_and_avoids_nan(predict_level0: bool, query_level0: bool):
    model = _make_model(
        use_rope=True,
        query_level0=query_level0,
        predict_level0=predict_level0,
        prefix_len=16,
    )
    p_eff = 8
    x_full, pad_full, x_trim, pad_trim = _build_inputs(
        model, batch_size=2, valid_lens=[8, 0], p_eff=p_eff, requires_grad=True
    )

    captured_masks = []
    orig_forward = model.transformer.blocks[0].attn.forward

    def spy_forward(x, attn_mask=None, rope_cos_sin=None, block_mask=None, **kwargs):
        captured_masks.append(attn_mask)
        return orig_forward(
            x, attn_mask=attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask, **kwargs
        )

    model.transformer.blocks[0].attn.forward = spy_forward
    try:
        logits = model(x_trim, prefix_pad_mask=pad_trim)
    finally:
        model.transformer.blocks[0].attn.forward = orig_forward

    assert len(captured_masks) == 1
    mask = captured_masks[0]
    t_target = sum(model.levels[model._lvl_first :])
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 1, p_eff + t_target, p_eff + t_target)
    assert bool(mask[1, 0, 0, 0].item()) is True
    assert not bool(mask[1, 0, 0, 1:p_eff].any().item())
    assert torch.isfinite(logits).all()
    assert int(torch.isnan(logits).sum().item()) == 0

    with torch.no_grad():
        logits_full = model(x_full.detach(), prefix_pad_mask=pad_full)
    assert torch.allclose(logits, logits_full, atol=1e-6, rtol=1e-5)

    logits.pow(2).mean().backward()
    assert torch.isfinite(x_trim.grad).all()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_bf16_autocast_with_qk_norm_and_trimmed_prefix():
    model = _make_model(
        use_rope=True,
        query_level0=False,
        predict_level0=True,
        use_qk_norm=True,
        prefix_len=16,
        attn_pattern="block_diagonal",
    )
    p_eff = 9
    x_full, pad_full, x_trim, pad_trim = _build_inputs(
        model, batch_size=2, valid_lens=[9, 0], p_eff=p_eff, requires_grad=True
    )

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        logits_trim = model(x_trim, prefix_pad_mask=pad_trim)
        loss = logits_trim.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(logits_trim).all()
    assert int(torch.isnan(logits_trim).sum().item()) == 0
    assert torch.isfinite(x_trim.grad).all()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name

    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
        logits_full = model(x_full.detach(), prefix_pad_mask=pad_full)
    assert torch.allclose(logits_trim.detach().float(), logits_full.float(), atol=1e-2, rtol=1e-2)


def test_self_attention_accepts_boolean_4d_sdpa_mask_and_kv_cache():
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        n_embd=16, n_head=4, n_kv_heads=2, bias=False, dropout=0.0, use_qk_norm=True
    )
    attn = SelfAttention(cfg).eval()

    x = torch.randn(2, 6, 16)
    bool_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)
    bool_mask[1, :, :, 3:] = False

    with torch.no_grad():
        out = attn(x, attn_mask=bool_mask)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out_bf16 = attn(x, attn_mask=bool_mask)
            out_pfx, kv = attn(x[:, :4], attn_mask=bool_mask[:, :, :4, :4], return_kv=True)
            out_dec = attn(x[:, 4:], attn_mask=bool_mask[:, :, 4:, :], past_kv=kv)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert int(torch.isnan(out).sum().item()) == 0
    assert torch.isfinite(out_bf16).all()
    assert out_pfx.shape == (2, 4, 16)
    assert out_dec.shape == (2, 2, 16)
    assert kv[0].dtype == kv[1].dtype


def test_state_dict_ctx_encoder_and_generate_kv_cache_compatibility():
    model = _make_model(
        use_rope=True,
        query_level0=False,
        use_ctx_encoder=True,
        prefix_len=12,
        attn_pattern="block_diagonal",
    )
    assert "attn_keep_mask" not in model.state_dict()
    assert "attn_mask_for_masking" in model.state_dict()

    t_target = sum(model.levels[model._lvl_first :])
    x_target = torch.randn(2, t_target, model.vqvae_local[0].embed_dim)
    prefix_ids = torch.randint(0, 64, (2, 8))
    pad_mask = torch.zeros(2, 8, dtype=torch.bool)
    pad_mask[1, 5:] = True
    with torch.no_grad():
        ctx_logits = model(x_target, prefix_ids=prefix_ids, prefix_pad_mask=pad_mask)
    assert ctx_logits.shape == (2, t_target, 32)
    assert torch.isfinite(ctx_logits).all()

    gen_model = _build_real_vqvae_model(prefix_len=12, levels=(2, 4, 8), predict_level0=True)
    prefix_tokens = torch.randint(1, 32, (2, 12))
    prefix_tokens[0, -4:] = 0
    with torch.no_grad():
        toks_kv = gen_model.generate(None, prefix_tokens, temperature=0.0, use_kv_cache=True)
        toks_nokv = gen_model.generate(None, prefix_tokens, temperature=0.0, use_kv_cache=False)
    assert toks_kv.shape == (2, 14)
    assert torch.equal(toks_kv, toks_nokv)

from types import SimpleNamespace
import pytest
import torch

from models.common import SelfAttention
from models.next_concept import NextConceptTransformer


def _make_model(
    *,
    use_rope: bool,
    query_level0: bool,
    predict_level0: bool = False,
    prefix_len: int = 16,
    levels: tuple[int, ...] = (2, 4, 8),
    attn_pattern: str = "block_causal",
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
        dropout=0.0,
        bias=False,
        vocab_size=head_vocabs,
        attn_pattern=attn_pattern,
        query_level0=query_level0,
        l0_autoregressive=False,
        prefix_len=prefix_len,
        use_rope=use_rope,
        rope_base=10000,
        use_level_encoding=True,
        use_flex_attention=False,
        predict_level0=predict_level0,
    )
    model = NextConceptTransformer(vqvae_stub, cfg)
    model.eval()
    return model


def _build_inputs(model: NextConceptTransformer, batch_size: int, valid_lens: list[int], p_eff: int):
    p_full = model.prefix_len
    t_target = sum(model.levels[model._lvl_first :])
    d_in = model.vqvae_local[0].embed_dim

    x_full = torch.randn(batch_size, p_full + t_target, d_in)
    pad_mask_full = torch.zeros(batch_size, p_full, dtype=torch.bool)
    for b, vlen in enumerate(valid_lens):
        if vlen < p_full:
            pad_mask_full[b, vlen:] = True

    x_trimmed = torch.cat([x_full[:, :p_eff, :], x_full[:, p_full:, :]], dim=1)
    pad_mask_trimmed = pad_mask_full[:, :p_eff]
    return x_full, pad_mask_full, x_trimmed, pad_mask_trimmed


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("query_level0", [False, True])
def test_right_pad_trimming_exact_logit_equivalence(use_rope: bool, query_level0: bool):
    model = _make_model(use_rope=use_rope, query_level0=query_level0, prefix_len=16)
    valid_lens = [10, 6]
    p_eff = 10
    x_full, pad_full, x_trim, pad_trim = _build_inputs(model, batch_size=2, valid_lens=valid_lens, p_eff=p_eff)

    with torch.no_grad():
        logits_full = model(x_full, prefix_pad_mask=pad_full)
        logits_trim_full_mask = model(x_trim, prefix_pad_mask=pad_full)
        logits_trim_short_mask = model(x_trim, prefix_pad_mask=pad_trim)

    assert torch.isfinite(logits_full).all()
    assert torch.allclose(logits_trim_full_mask, logits_full, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_trim_short_mask, logits_full, atol=1e-6, rtol=1e-5)


def test_right_pad_trimming_with_predict_level0():
    model = _make_model(use_rope=True, query_level0=False, predict_level0=True, prefix_len=16)
    valid_lens = [11, 5]
    p_eff = 11
    x_full, pad_full, x_trim, pad_trim = _build_inputs(model, batch_size=2, valid_lens=valid_lens, p_eff=p_eff)

    with torch.no_grad():
        logits_full = model(x_full, prefix_pad_mask=pad_full)
        logits_trim = model(x_trim, prefix_pad_mask=pad_trim)

    assert torch.isfinite(logits_full).all()
    assert torch.allclose(logits_trim, logits_full, atol=1e-6, rtol=1e-5)


def test_partial_padding_ignores_trailing_padded_values():
    model = _make_model(use_rope=True, query_level0=False, prefix_len=16)
    p_eff = 10
    x_full, pad_full, x_trim, pad_trim = _build_inputs(model, batch_size=2, valid_lens=[10, 4], p_eff=p_eff)

    x_trim_perturbed = x_trim.clone()
    x_trim_perturbed[1, 4:p_eff, :] += 50.0

    with torch.no_grad():
        logits_a = model(x_trim, prefix_pad_mask=pad_trim)
        logits_b = model(x_trim_perturbed, prefix_pad_mask=pad_trim)
        logits_full = model(x_full, prefix_pad_mask=pad_full)

    assert torch.allclose(logits_a, logits_full, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5)


def test_all_pad_row_unmasks_col0_and_avoids_nan():
    model = _make_model(use_rope=True, query_level0=True, prefix_len=16)
    p_eff = 8
    x_full, pad_full, x_trim, pad_trim = _build_inputs(model, batch_size=2, valid_lens=[8, 0], p_eff=p_eff)

    captured_masks = []
    orig_forward = model.transformer.blocks[0].attn.forward

    def spy_forward(x, attn_mask=None, rope_cos_sin=None, block_mask=None):
        captured_masks.append(attn_mask)
        return orig_forward(x, attn_mask=attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask)

    model.transformer.blocks[0].attn.forward = spy_forward
    try:
        with torch.no_grad():
            logits = model(x_trim, prefix_pad_mask=pad_trim)
    finally:
        model.transformer.blocks[0].attn.forward = orig_forward

    assert len(captured_masks) == 1
    mask = captured_masks[0]
    t_target = sum(model.levels[model._lvl_first :])
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 1, p_eff + t_target, p_eff + t_target)
    # Row 1 is all-pad: col0 is forced True so prefix queries have at least one valid key.
    assert bool(mask[1, 0, 0, 0].item()) is True
    assert not bool(mask[1, 0, 0, 1:p_eff].any().item())
    assert torch.isfinite(logits).all()
    assert int(torch.isnan(logits).sum().item()) == 0


def test_self_attention_accepts_boolean_4d_sdpa_mask():
    torch.manual_seed(0)
    cfg = SimpleNamespace(n_embd=16, n_head=2, n_kv_heads=2, bias=False, dropout=0.0, use_qk_norm=False)
    attn = SelfAttention(cfg)
    attn.eval()

    x = torch.randn(2, 6, 16)
    bool_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)
    bool_mask[1, :, :, 3:] = False

    with torch.no_grad():
        out = attn(x, attn_mask=bool_mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert int(torch.isnan(out).sum().item()) == 0

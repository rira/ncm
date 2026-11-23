import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import Encoder, SelfAttention
from models.vqvae import VQVAE


def _make_vqvae(
    *,
    pad_token_id=None,
    pre_quant_norm_type="group",
    enc_layers=2,
    enc_attn_window=0,
    enc_conv_layers=0,
    compression_factor=1,
    context_length=16,
):
    levels = [4, 16] if compression_factor == 1 else [2, 4]
    vq_cfg = OmegaConf.create(
        {
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.99,
            "commitment_cost": 0.25,
            "levels": {
                "use_manual_levels": True,
                "manual_levels": levels,
            },
        }
    )
    kwargs = dict(
        in_vocab_size=32,
        context_length=context_length,
        embed_dim=16,
        vqvae_vocab_size=[8, 8],
        n_layers=2,
        vector_quantizer_config=vq_cfg,
        compression_factor=compression_factor,
        dropout=0.0,
        pre_quant_groupnorm=4,
        pre_quant_dropout=0.0,
        use_qk_norm=True,
        enc_layers=enc_layers,
        enc_attn_window=enc_attn_window,
        enc_conv_layers=enc_conv_layers,
        dec_layers=1,
        pre_quant_norm_type=pre_quant_norm_type,
    )
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return VQVAE(**kwargs)


@pytest.mark.parametrize("attn_window", [0, 3])
def test_self_attention_default_unmasked_bit_identical(attn_window):
    torch.manual_seed(0)
    cfg = type(
        "Config",
        (object,),
        {
            "n_embd": 16,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "use_qk_norm": True,
            "attn_window": attn_window,
        },
    )
    attn = SelfAttention(cfg).eval()
    x = torch.randn(2, 12, 16)
    out_default = attn(x)
    out_none = attn(x, key_padding_mask=None)
    out_no_pad = attn(x, key_padding_mask=torch.zeros(2, 12, dtype=torch.bool))
    assert torch.equal(out_default, out_none)
    assert torch.allclose(out_default, out_no_pad, atol=1e-6, rtol=0.0)


def test_vqvae_pad_token_id_none_default_preserves_unmasked_forward():
    torch.manual_seed(1)
    model_default = _make_vqvae(pad_token_id=None, enc_attn_window=3).eval()
    assert model_default.pad_token_id is None
    assert model_default.encoder.pad_token_id is None

    model_explicit_none = _make_vqvae(enc_attn_window=3).eval()
    model_explicit_none.load_state_dict(model_default.state_dict(), strict=True)

    model_unpadded_masked = _make_vqvae(pad_token_id=0, enc_attn_window=3).eval()
    model_unpadded_masked.load_state_dict(model_default.state_dict(), strict=True)

    x_no_pads = torch.randint(1, 32, (2, 16))
    with torch.no_grad():
        logits_def, vq_def, _, _, _ = model_default(x_no_pads)
        z_def = model_default.encode_z(x_no_pads)
        logits_none, vq_none, _, _, _ = model_explicit_none(x_no_pads)
        z_none = model_explicit_none.encode_z(x_no_pads)
        logits_msk, vq_msk, _, _, _ = model_unpadded_masked(x_no_pads)
        z_msk = model_unpadded_masked.encode_z(x_no_pads)

    assert torch.equal(z_def, z_none)
    assert torch.equal(logits_def, logits_none)
    assert torch.equal(vq_def, vq_none)
    assert torch.allclose(z_def, z_msk, atol=1e-6, rtol=0.0)
    assert torch.allclose(logits_def, logits_msk, atol=1e-6, rtol=0.0)
    assert torch.allclose(vq_def, vq_msk, atol=1e-6, rtol=0.0)


def test_encoder_forward_nopos_isolates_pad_tokens():
    torch.manual_seed(4)
    pad_id = 0
    enc = Encoder(
        in_vocab_size=32,
        context_length=16,
        embed_dim=16,
        n_layers=2,
        compression_factor=1,
        dropout=0.0,
        enc_layers=2,
        use_qk_norm=True,
        enc_conv_layers=1,
        use_pos_emb=False,
        pad_token_id=pad_id,
    ).eval()
    x = torch.tensor(
        [[2, 5, 9, 13, pad_id, pad_id, pad_id, pad_id]],
        dtype=torch.long,
    )
    real_mask = x != pad_id
    pad_mask = x == pad_id
    with torch.no_grad():
        out_before = enc.forward_nopos(x)
        enc.wte.weight[pad_id].mul_(100.0).add_(25.0)
        out_after = enc.forward_nopos(x)

    assert (out_after[real_mask] - out_before[real_mask]).abs().max().item() < 1e-5
    assert torch.equal(out_after[pad_mask], torch.zeros_like(out_after[pad_mask]))


@pytest.mark.parametrize("norm_type", ["group", "layer"])
@pytest.mark.parametrize("enc_attn_window", [0, 3])
@pytest.mark.parametrize("enc_conv_layers", [0, 1])
def test_mixed_padded_sequences_isolate_real_tokens(
    norm_type, enc_attn_window, enc_conv_layers
):
    torch.manual_seed(2)
    pad_id = 0
    model = _make_vqvae(
        pad_token_id=pad_id,
        pre_quant_norm_type=norm_type,
        enc_layers=2,
        enc_attn_window=enc_attn_window,
        enc_conv_layers=enc_conv_layers,
    ).eval()

    x = torch.tensor(
        [
            [3, 7, 11, 15, 19, 23, 27, 31, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id],
            [5, 9, 13, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id],
        ],
        dtype=torch.long,
    )
    real_mask = x != pad_id
    pad_mask = x == pad_id

    with torch.no_grad():
        z_before = model.encode_z(x)
        model.encoder.wte.weight[pad_id].mul_(100.0).add_(25.0)
        z_after = model.encode_z(x)

    max_diff = (z_after[real_mask] - z_before[real_mask]).abs().max().item()
    assert max_diff < 1e-5
    assert torch.equal(z_before[pad_mask], torch.zeros_like(z_before[pad_mask]))
    assert torch.equal(z_after[pad_mask], torch.zeros_like(z_after[pad_mask]))


@pytest.mark.parametrize("norm_type", ["group", "layer"])
@pytest.mark.parametrize("compression_factor", [1, 4])
def test_all_pad_batch_produces_finite_outputs_and_exact_zero_latents(
    norm_type, compression_factor
):
    torch.manual_seed(3)
    pad_id = 0
    model = _make_vqvae(
        pad_token_id=pad_id,
        pre_quant_norm_type=norm_type,
        enc_layers=2,
        enc_attn_window=3,
        enc_conv_layers=1,
        compression_factor=compression_factor,
    ).train()

    x = torch.full((2, 16), pad_id, dtype=torch.long)
    z = model._encode_pre_quant(x)
    assert torch.equal(z, torch.zeros_like(z))

    logits, vq_loss, _, _, _ = model(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(vq_loss)

    model.zero_grad()
    (logits.sum() + vq_loss).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)

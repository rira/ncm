import copy
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.next_concept import _TiedCodebookHead  # noqa: E402


def test_learn_scale_false_registers_non_persistent_buffer():
    codebook = torch.randn(32, 8)
    head = _TiedCodebookHead(16, codebook, learn_scale=False)

    buffers = dict(head.named_buffers())
    params = dict(head.named_parameters())
    assert "log_inv_T" in buffers
    assert "log_inv_T" not in params
    assert "log_inv_T" not in head.state_dict()

    head_bf16 = copy.deepcopy(head).to(torch.bfloat16)
    assert head_bf16.log_inv_T.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in head_bf16.parameters())
    assert all(b.dtype == torch.bfloat16 for b in head_bf16.buffers())

    x_bf16 = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    out_bf16 = head_bf16(x_bf16)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.isfinite(out_bf16).all()

    head_meta = copy.deepcopy(head).to("meta")
    assert head_meta.log_inv_T.device.type == "meta"
    assert all(p.device.type == "meta" for p in head_meta.parameters())
    assert all(b.device.type == "meta" for b in head_meta.buffers())


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp32_distance_and_bf16_forward_backward_parity(trainable, use_bias):
    torch.manual_seed(0)
    B, T, n_embd, V, d = 2, 5, 32, 24, 16
    codebook = torch.randn(V, d)

    head_ref = _TiedCodebookHead(n_embd, codebook, learn_scale=True, trainable=trainable, use_bias=use_bias)
    with torch.no_grad():
        head_ref.log_inv_T.fill_(0.5)
        if use_bias:
            head_ref.bias.normal_(std=0.1)

    head_ac = copy.deepcopy(head_ref)
    head_bf16 = copy.deepcopy(head_ref).to(torch.bfloat16)

    x_base = torch.randn(B, T, n_embd)
    targets = torch.randint(0, V, (B * T,))

    x_ref = x_base.clone().requires_grad_(True)
    out_ref = head_ref(x_ref)
    assert out_ref.dtype == torch.float32

    z_ref = F.linear(x_ref, head_ref.proj.weight)
    neg_sq_dist = -((z_ref.unsqueeze(-2) - head_ref.E.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
    expected_logits = (neg_sq_dist / head_ref.d_scale) * torch.exp(head_ref.log_inv_T)
    if use_bias:
        expected_logits = expected_logits + head_ref.bias
    torch.testing.assert_close(
        F.log_softmax(out_ref, dim=-1),
        F.log_softmax(expected_logits, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )

    F.cross_entropy(out_ref.reshape(-1, V), targets).backward()

    x_ac = x_base.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_ac = head_ac(x_ac)
    assert out_ac.dtype == torch.bfloat16
    torch.testing.assert_close(out_ac.float(), out_ref.detach(), atol=5e-2, rtol=5e-2)
    F.cross_entropy(out_ac.float().reshape(-1, V), targets).backward()

    x_bf16 = x_base.to(torch.bfloat16).requires_grad_(True)
    out_bf16 = head_bf16(x_bf16)
    assert out_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(out_bf16.float(), out_ref.detach(), atol=5e-2, rtol=5e-2)
    F.cross_entropy(out_bf16.float().reshape(-1, V), targets).backward()

    for name, p_ref in head_ref.named_parameters():
        p_ac = dict(head_ac.named_parameters())[name]
        p_bf16 = dict(head_bf16.named_parameters())[name]
        assert p_ac.grad is not None and torch.isfinite(p_ac.grad).all()
        assert p_bf16.grad is not None and torch.isfinite(p_bf16.grad).all()
        torch.testing.assert_close(p_ac.grad, p_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(p_bf16.grad.float(), p_ref.grad, atol=3e-2, rtol=3e-2)

    torch.testing.assert_close(x_ac.grad, x_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x_bf16.grad.float(), x_ref.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("extreme, clamped", [(100.0, 6.0), (-100.0, -10.0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_inv_T_extreme_values_clamped_and_finite(extreme, clamped, dtype):
    torch.manual_seed(1)
    codebook = torch.randn(16, 8)
    head_extreme = _TiedCodebookHead(16, codebook, learn_scale=True, trainable=True, use_bias=True).to(dtype)
    head_clamped = copy.deepcopy(head_extreme)

    with torch.no_grad():
        head_extreme.log_inv_T.fill_(extreme)
        head_clamped.log_inv_T.fill_(clamped)

    x = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
    out_extreme = head_extreme(x)
    out_clamped = head_clamped(x.detach())

    assert out_extreme.dtype == dtype
    assert torch.isfinite(out_extreme).all()
    torch.testing.assert_close(out_extreme, out_clamped)

    out_extreme.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in head_extreme.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert head_extreme.log_inv_T.grad.item() == 0.0

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xpu_compat  # noqa: E402
from models.common import SelfAttention  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.quant import MultiscaleResidualVectorQuantizer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402


def _make_fake_xpu(device_names: list[str], available: bool = True, capabilities: dict[int, dict] | None = None):
    fake_xpu = MagicMock()
    fake_xpu.is_available.return_value = available
    fake_xpu.device_count.return_value = len(device_names) if available else 0
    fake_xpu.get_device_name.side_effect = lambda idx: device_names[idx]
    caps = capabilities or {}
    fake_xpu.get_device_capability.side_effect = lambda dev: caps.get(
        torch.device(dev).index if not isinstance(dev, int) else dev,
        {"has_fp64": False},
    )
    return fake_xpu


class TestXpuDiscovery:
    def test_prefers_discrete_arc_over_uhd_at_index_1(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            [
                "Intel(R) UHD Graphics 770",
                "Intel(R) Arc(TM) A770 Graphics",
            ]
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)

        assert xpu_compat.is_xpu_available() is True
        assert xpu_compat.get_discrete_xpu_indices() == [1]
        assert xpu_compat.select_best_xpu_index() == 1

    def test_prefers_discrete_arc_when_at_index_0(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            [
                "Intel(R) Arc(TM) A770 Graphics",
                "Intel(R) UHD Graphics 770",
            ]
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)

        assert xpu_compat.get_discrete_xpu_indices() == [0]
        assert xpu_compat.select_best_xpu_index() == 0

    def test_fallback_non_uhd_and_all_indices(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            [
                "Intel(R) UHD Graphics 770",
                "Intel(R) Data Center GPU Max 1100",
            ]
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        assert xpu_compat.get_discrete_xpu_indices() == [1]

        uhd_only = _make_fake_xpu(["Intel(R) UHD Graphics 770"])
        monkeypatch.setattr(torch, "xpu", uhd_only, raising=False)
        assert xpu_compat.get_discrete_xpu_indices() == [0]
        assert xpu_compat.select_best_xpu_index() == 0

    def test_unavailable_xpu_returns_empty(self, monkeypatch):
        fake_xpu = _make_fake_xpu([], available=False)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        assert xpu_compat.is_xpu_available() is False
        assert xpu_compat.get_discrete_xpu_indices() == []


class TestResolveDevice:
    def test_env_override_xpu_selects_best_arc_index(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            [
                "Intel(R) UHD Graphics 770",
                "Intel(R) Arc(TM) A770 Graphics",
            ]
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setenv("NCM_DEVICE", "xpu")

        dev = xpu_compat.resolve_device()
        assert dev == torch.device("xpu:1")
        fake_xpu.set_device.assert_called_once_with(torch.device("xpu:1"))

    def test_env_override_explicit_xpu_0_and_cpu(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            [
                "Intel(R) UHD Graphics 770",
                "Intel(R) Arc(TM) A770 Graphics",
            ]
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)

        monkeypatch.setenv("NCM_DEVICE", "xpu:0")
        dev0 = xpu_compat.resolve_device()
        assert dev0 == torch.device("xpu:0")
        fake_xpu.set_device.assert_called_with(torch.device("xpu:0"))

        monkeypatch.setenv("NCM_DEVICE", "cpu")
        dev_cpu = xpu_compat.resolve_device()
        assert dev_cpu == torch.device("cpu")

    def test_auto_detection_priority_xpu_cuda_mps_cpu(self, monkeypatch):
        monkeypatch.delenv("NCM_DEVICE", raising=False)
        fake_xpu = _make_fake_xpu(["Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        set_cuda = MagicMock()
        monkeypatch.setattr(torch.cuda, "set_device", set_cuda)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        assert xpu_compat.resolve_device() == torch.device("xpu:1")
        set_cuda.assert_not_called()

        fake_xpu.is_available.return_value = False
        dev_cuda = xpu_compat.resolve_device()
        assert dev_cuda == torch.device("cuda:0")
        set_cuda.assert_called_once_with(torch.device("cuda:0"))

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            assert xpu_compat.resolve_device() == torch.device("mps")
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        assert xpu_compat.resolve_device() == torch.device("cpu")


class TestWorkerDeviceConfig:
    def test_get_worker_device_ids_xpu_and_cuda(self, monkeypatch):
        monkeypatch.delenv("NCM_DEVICE", raising=False)
        fake_xpu = _make_fake_xpu(["Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert xpu_compat.get_worker_device_ids(None) == ("xpu", ["1"])
        assert xpu_compat.get_worker_device_ids("0,1") == ("xpu", ["0", "1"])

        fake_xpu.is_available.return_value = False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert xpu_compat.get_worker_device_ids(None) == ("cuda", ["0", "1"])
        assert xpu_compat.get_worker_device_ids("1") == ("cuda", ["1"])

    def test_configure_worker_env_xpu_preserves_ze_affinity_mask(self):
        base = {"PATH": "/usr/bin"}
        env_xpu = xpu_compat.configure_worker_env(base, "xpu", "1")
        assert env_xpu["NCM_DEVICE"] == "xpu:1"
        assert "ZE_AFFINITY_MASK" not in env_xpu
        assert "NCM_DEVICE" not in base

        base_with_ze = {"ZE_AFFINITY_MASK": "0.1"}
        env_xpu_kept = xpu_compat.configure_worker_env(base_with_ze, "xpu", "1")
        assert env_xpu_kept["NCM_DEVICE"] == "xpu:1"
        assert env_xpu_kept["ZE_AFFINITY_MASK"] == "0.1"

    def test_configure_worker_env_cuda(self):
        base = {"PATH": "/usr/bin"}
        env_cuda = xpu_compat.configure_worker_env(base, "cuda", "2")
        assert env_cuda["CUDA_VISIBLE_DEVICES"] == "2"
        assert env_cuda["NCM_DEVICE"] == "cuda:0"


class TestPrecisionAndDiagnostics:
    def test_autocast_context_and_matmul_precision(self, monkeypatch):
        with xpu_compat.autocast_context("cpu", dtype=torch.bfloat16, enabled=False) as ctx:
            assert ctx is None
        with xpu_compat.autocast_context("cpu", dtype=torch.bfloat16, enabled=True):
            a = torch.randn(2, 2)
            assert (a @ a).shape == (2, 2)

        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=False)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        ctx_unavail = xpu_compat.autocast_context("xpu:0", dtype=torch.bfloat16, enabled=True)
        assert isinstance(ctx_unavail, contextlib.nullcontext)

        called = []
        monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda mode: called.append(mode))
        xpu_compat.configure_matmul_precision("xpu:0")
        xpu_compat.configure_matmul_precision("cuda:0")
        assert called == ["high", "high"]

    def test_supports_fp64(self, monkeypatch):
        fake_xpu = _make_fake_xpu(
            ["Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770 Graphics"],
            available=True,
            capabilities={0: {"has_fp64": True}, 1: {"has_fp64": False}},
        )
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)

        assert xpu_compat.supports_fp64("xpu:1") is False
        assert xpu_compat.supports_fp64("xpu:0") is True
        assert xpu_compat.supports_fp64("cuda:0") is True
        assert xpu_compat.supports_fp64("cpu") is True
        assert xpu_compat.supports_fp64("mps") is False

    def test_check_ollama_vram_notice(self):
        payload = json.dumps({"models": [{"name": "qwen2.5:14b", "size_vram": 8 * 1024**3}]}).encode("utf-8")

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            notice = xpu_compat.check_ollama_vram_notice()
            assert notice is not None
            assert "qwen2.5:14b" in notice

        empty_payload = json.dumps({"models": []}).encode("utf-8")

        class _EmptyResp(_FakeResp):
            def read(self):
                return empty_payload

        with patch("urllib.request.urlopen", return_value=_EmptyResp()):
            assert xpu_compat.check_ollama_vram_notice() is None

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(ConnectionRefusedError("refused"))):
            assert xpu_compat.check_ollama_vram_notice() is None

    def test_synchronize_empty_cache_seed_and_vram_stats(self, monkeypatch):
        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        fake_xpu.memory_allocated.return_value = 64 * 1024 * 1024
        fake_xpu.memory_reserved.return_value = 128 * 1024 * 1024
        fake_xpu.max_memory_allocated.return_value = 96 * 1024 * 1024
        fake_xpu.get_device_properties.return_value = SimpleNamespace(total_memory=16384 * 1024 * 1024)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        xpu_compat.synchronize("xpu:0")
        fake_xpu.synchronize.assert_called_once_with(torch.device("xpu:0"))
        xpu_compat.empty_cache("xpu:0")
        fake_xpu.empty_cache.assert_called_once()
        xpu_compat.seed_everything(42)
        fake_xpu.manual_seed_all.assert_called_once_with(42)

        stats = xpu_compat.get_vram_stats_mb("xpu:0")
        assert stats["allocated_mb"] == pytest.approx(64.0)
        assert stats["reserved_mb"] == pytest.approx(128.0)
        assert stats["max_allocated_mb"] == pytest.approx(96.0)
        assert stats["total_mb"] == pytest.approx(16384.0)

        cpu_stats = xpu_compat.get_vram_stats_mb("cpu")
        assert cpu_stats == {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "total_mb": 0.0}

    def test_configure_vram_cap_cpu_and_disabled(self, monkeypatch):
        res_cpu = xpu_compat.configure_vram_cap("cpu", max_reserved_mb=11800, memory_fraction=0.86)
        assert res_cpu == {"applied": False, "fraction": 0.0, "max_reserved_mb": 11800.0}

        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        res_zero = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.0)
        assert res_zero == {"applied": False, "fraction": 0.0, "max_reserved_mb": 11800.0}
        fake_xpu.set_per_process_memory_fraction.assert_not_called()

    def test_configure_vram_cap_xpu_direct_fallback_and_clamping(self, monkeypatch):
        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)

        res = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.86)
        assert res == {"applied": True, "fraction": pytest.approx(0.86), "max_reserved_mb": 11800.0}
        fake_xpu.set_per_process_memory_fraction.assert_called_once_with(0.86, torch.device("xpu:0"))

        res_low = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.01)
        assert res_low["applied"] is True
        assert res_low["fraction"] == pytest.approx(0.1)

        res_high = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=1.5)
        assert res_high["applied"] is True
        assert res_high["fraction"] == pytest.approx(0.98)

        # Fallback to fake_xpu.memory.set_per_process_memory_fraction when direct attr is absent
        fake_xpu_mem = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        del fake_xpu_mem.set_per_process_memory_fraction
        monkeypatch.setattr(torch, "xpu", fake_xpu_mem, raising=False)
        res_mem = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.86)
        assert res_mem == {"applied": True, "fraction": pytest.approx(0.86), "max_reserved_mb": 11800.0}
        fake_xpu_mem.memory.set_per_process_memory_fraction.assert_called_once_with(0.86, torch.device("xpu:0"))

        # Fallback to int device index when TypeError is raised for torch.device
        def _int_only_setter(frac, dev_arg):
            if isinstance(dev_arg, torch.device):
                raise TypeError("expected int device index")
            return dev_arg

        fake_xpu_int = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        fake_xpu_int.set_per_process_memory_fraction.side_effect = _int_only_setter
        monkeypatch.setattr(torch, "xpu", fake_xpu_int, raising=False)
        res_int = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.86)
        assert res_int == {"applied": True, "fraction": pytest.approx(0.86), "max_reserved_mb": 11800.0}
        assert fake_xpu_int.set_per_process_memory_fraction.call_args_list[-1].args == (0.86, 0)

    def test_configure_vram_cap_graceful_error_fallback(self, monkeypatch):
        for exc in (AttributeError("missing"), RuntimeError("unsupported")):
            fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
            fake_xpu.set_per_process_memory_fraction.side_effect = exc
            monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
            res = xpu_compat.configure_vram_cap("xpu:0", max_reserved_mb=11800, memory_fraction=0.86)
            assert res == {"applied": False, "fraction": 0.0, "max_reserved_mb": 11800.0}

    def test_trim_vram_cache_if_needed(self, monkeypatch):
        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert xpu_compat.trim_vram_cache_if_needed("cpu", 11800.0) is False
        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 0.0) is False
        fake_xpu.empty_cache.assert_not_called()

        fake_xpu.memory_reserved.return_value = 10000 * 1024 * 1024
        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 11800.0) is False
        fake_xpu.empty_cache.assert_not_called()

        fake_xpu.memory_reserved.return_value = 12200 * 1024 * 1024
        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 11800.0) is True
        fake_xpu.empty_cache.assert_called_once()

    def test_trim_vram_cache_cooldown_and_slack(self, monkeypatch):
        fake_xpu = _make_fake_xpu(["Intel(R) Arc(TM) A770 Graphics"], available=True)
        monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(xpu_compat, "_LAST_VRAM_TRIM_TS", 0.0, raising=False)

        fake_xpu.memory_reserved.return_value = 12600 * 1024 * 1024
        fake_xpu.memory_allocated.return_value = 12300 * 1024 * 1024
        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 12400.0, min_interval_sec=60.0) is False
        fake_xpu.empty_cache.assert_not_called()

        fake_xpu.memory_allocated.return_value = 10000 * 1024 * 1024
        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 12400.0, min_interval_sec=60.0) is True
        fake_xpu.empty_cache.assert_called_once()

        assert xpu_compat.trim_vram_cache_if_needed("xpu:0", 12400.0, min_interval_sec=60.0) is False
        fake_xpu.empty_cache.assert_called_once()

    def test_get_process_commit_gb(self, monkeypatch):
        monkeypatch.setattr(xpu_compat.os, "name", "posix")
        assert xpu_compat.get_process_commit_gb() == 0.0

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only process commit check")
    def test_get_process_commit_gb_real_windows(self):
        commit_gb = xpu_compat.get_process_commit_gb()
        assert isinstance(commit_gb, float)
        assert math.isfinite(commit_gb)
        assert 0.01 < commit_gb < 500.0

    def test_lower_process_priority_for_desktop(self, monkeypatch):
        import ctypes

        monkeypatch.setattr(xpu_compat.os, "name", "posix")
        assert xpu_compat.lower_process_priority_for_desktop() is False

        kernel32 = MagicMock()
        kernel32.GetCurrentProcess.return_value = 12345
        kernel32.SetPriorityClass.return_value = 1
        fake_windll = SimpleNamespace(kernel32=kernel32)
        monkeypatch.setattr(xpu_compat.os, "name", "nt")
        monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)

        assert xpu_compat.lower_process_priority_for_desktop() is True
        assert kernel32.GetCurrentProcess.restype == ctypes.c_void_p
        assert kernel32.GetCurrentProcess.argtypes == []
        assert kernel32.SetPriorityClass.argtypes == [ctypes.c_void_p, ctypes.c_uint32]
        assert kernel32.SetPriorityClass.restype == ctypes.c_int
        kernel32.GetCurrentProcess.assert_called_once_with()
        kernel32.SetPriorityClass.assert_called_once_with(12345, 0x00004000)

        kernel32.SetPriorityClass.side_effect = OSError("access denied")
        assert xpu_compat.lower_process_priority_for_desktop() is False

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only handle check")
    def test_lower_process_priority_for_desktop_real_windows(self):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        get_cur = kernel32.GetCurrentProcess
        get_cur.argtypes = []
        get_cur.restype = ctypes.c_void_p
        get_prio = kernel32.GetPriorityClass
        get_prio.argtypes = [ctypes.c_void_p]
        get_prio.restype = ctypes.c_uint32
        set_prio = kernel32.SetPriorityClass
        set_prio.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        set_prio.restype = ctypes.c_int

        handle = get_cur()
        orig_prio = get_prio(handle) or 0x20
        try:
            ok = xpu_compat.lower_process_priority_for_desktop()
            assert ok is True
            assert get_prio(handle) == 0x00004000
        finally:
            set_prio(handle, orig_prio)


class TestQuantAndModelCore:
    def test_idx_to_zhats_no_inplace_aliasing(self):
        levels_cfg = SimpleNamespace(use_manual_levels=True, manual_levels=[1, 2, 4])
        vq = MultiscaleResidualVectorQuantizer(
            vqvae_vocab_size=[8, 8, 8],
            embed_dim=16,
            levels=levels_cfg,
            quant_resi={"enabled": True, "ratio": 0.5, "share_mode": 0},
        )
        vq.eval()
        idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)

        z_hats, z_levels = vq.idx_to_zhats(idx, return_zlevels=True)
        assert len(z_hats) == 3
        assert len(z_levels) == 3
        assert len({id(t) for t in z_hats}) == 3
        assert id(z_levels[-1]) != id(z_hats[-1])
        assert not torch.allclose(z_hats[0], z_hats[-1])

        ref_final = vq.idx_to_z_q(idx)
        assert torch.allclose(z_hats[-1], ref_final, atol=1e-6)

        snapshot = z_hats[1].clone()
        z_hats[0].zero_()
        assert torch.allclose(z_hats[1], snapshot)

    def test_self_attention_aligns_float_mask_dtype_to_query(self):
        cfg = SimpleNamespace(n_embd=16, n_head=2, bias=True, dropout=0.0, use_qk_norm=True)
        attn = SelfAttention(cfg).eval()
        x = torch.randn(2, 4, 16)
        mask_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)

        observed = {}
        orig_sdpa = F.scaled_dot_product_attention

        def _spy_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            observed["q_dtype"] = q.dtype
            observed["k_dtype"] = k.dtype
            observed["v_dtype"] = v.dtype
            observed["mask_dtype"] = attn_mask.dtype if attn_mask is not None else None
            return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=_spy_sdpa):
            with xpu_compat.autocast_context("cpu", dtype=torch.bfloat16, enabled=True):
                out = attn(x, attn_mask=mask_fp32)

        assert out.shape == (2, 4, 16)
        assert observed["q_dtype"] == observed["k_dtype"] == observed["v_dtype"] == torch.bfloat16
        assert observed["mask_dtype"] == torch.bfloat16

    def test_next_concept_transformer_forward_and_generate_cpu(self):
        torch.manual_seed(0)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=32,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[8, 8, 8],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).eval()

        model_cfg = OmegaConf.create({
            "n_layer": 2,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "context_length": 8,
            "prefix_len": 2,
            "raw_token_prefix": True,
            "prefix_pad_token_id": 0,
            "vocab_size": [8, 8, 8],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).eval()

        B = 2
        x = torch.randn(B, 2 + 1 + 2 + 4, 16)
        pad_mask = torch.tensor([[False, True], [False, False]])
        with xpu_compat.autocast_context("cpu", dtype=torch.bfloat16, enabled=True):
            logits = net(x, prefix_pad_mask=pad_mask)
        assert logits.shape == (B, 7, 8)

        prefix_tokens = torch.tensor([[3, 0], [5, 7]], dtype=torch.long)
        with xpu_compat.autocast_context("cpu", dtype=torch.bfloat16, enabled=True):
            gen = net.generate(None, prefix_tokens, temperature=0.8, top_k=50, top_p=0.95)
        assert gen.shape == (B, 7)

        l0 = torch.randint(0, 8, (B, 1))
        gen_kv = net.generate(l0, prefix_tokens, temperature=0.0, top_k=0, top_p=1.0, use_kv_cache=False)
        assert gen_kv.shape == (B, 7)

        net.use_flex_attention = True
        assert net._should_use_flex_attention(torch.device("xpu:1")) is False
        with patch.object(net, "_should_use_flex_attention", return_value=False):
            with patch.object(net, "_get_flex_block_mask", side_effect=AssertionError("flex_attention must not run")):
                fallback_logits = net(x)
        assert fallback_logits.shape == (B, 7, 8)

    def test_next_concept_transformer_kv_cache_and_tokenizer_cpu(self):
        from tokenization.sudoku_tokenizer import SudokuTokenizer
        from tokenization.vqvae_tokenizer import VQVAETokenizer

        torch.manual_seed(1)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=12,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[8, 8, 8],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).eval()

        model_cfg = OmegaConf.create({
            "n_layer": 2,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "context_length": 8,
            "prefix_len": 2,
            "raw_token_prefix": True,
            "vocab_size": [8, 8],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": False,
            "tie_heads_to_codebook": False,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).eval()
        B = 2
        l0 = torch.randint(0, 8, (B, 1))
        prefix_tokens = torch.randint(0, 12, (B, 2))
        out_eager = net.generate(l0, prefix_tokens, temperature=0.0, top_k=0, top_p=1.0, use_kv_cache=False)
        out_kv = net.generate(l0, prefix_tokens, temperature=0.0, top_k=0, top_p=1.0, use_kv_cache=True)
        assert out_eager.shape == (B, 7)
        assert torch.equal(out_eager, out_kv)

        tk = VQVAETokenizer(SudokuTokenizer(), vqvae)
        encoded = tk.encode("1234")
        assert len(encoded) == 7
        decoded = tk.decode(encoded)
        assert len(decoded) == 4
        reconstructed = tk.reconstruct("1234")
        assert len(reconstructed) == 4


def _resolve_test_device(device_str: str) -> torch.device:
    if device_str.startswith("xpu") and not xpu_compat.is_xpu_available():
        pytest.skip("XPU device not available")
    return torch.device(device_str)


class TestXpuBoolMaskModelsRegression:
    @pytest.mark.parametrize("device_str", ["cpu", "xpu:0"])
    @pytest.mark.parametrize("n_samples,vocab_size", [(512, 4096), (256, 64)])
    def test_kmeans_init_level_empty_and_populated_clusters(self, device_str, n_samples, vocab_size):
        device = _resolve_test_device(device_str)
        torch.manual_seed(123)
        levels_cfg = SimpleNamespace(use_manual_levels=True, manual_levels=[1])
        vq = MultiscaleResidualVectorQuantizer(
            vqvae_vocab_size=[vocab_size],
            embed_dim=16,
            levels=levels_cfg,
        ).to(device)

        samples = torch.randn(n_samples, 16, device=device)
        vq.kmeans_init_level(0, samples, n_iters=5)
        xpu_compat.synchronize(device)

        w = vq.embeddings[0].weight
        ema_w = vq.ema_w_0
        ema_cs = vq.ema_cluster_size_0

        assert w.shape == (vocab_size, 16)
        assert torch.isfinite(w).all()
        assert torch.isfinite(ema_w).all()
        assert torch.allclose(w.float(), ema_w.float())
        assert torch.equal(ema_cs, torch.ones_like(ema_cs))

    @pytest.mark.parametrize("device_str", ["cpu", "xpu:0"])
    def test_restart_dead_codes_preserves_live_rows_and_replaces_dead_rows(self, device_str):
        device = _resolve_test_device(device_str)
        torch.manual_seed(456)
        levels_cfg = SimpleNamespace(use_manual_levels=True, manual_levels=[4])
        vq = MultiscaleResidualVectorQuantizer(
            vqvae_vocab_size=[32],
            embed_dim=16,
            levels=levels_cfg,
            restart_threshold=0.5,
        ).to(device)

        emb_i = vq.embeddings[0]
        ema_w = vq.ema_w_0
        ema_cs = vq.ema_cluster_size_0

        ema_cs[:12] = 0.1
        ema_cs[12:] = 2.5

        orig_weight = emb_i.weight.data.clone()
        orig_ema_w = ema_w.clone()
        orig_ema_cs = ema_cs.clone()

        flat_input = torch.randn(24, 16, device=device)
        vq._restart_dead_codes(flat_input, ema_cs, ema_w, emb_i)
        xpu_compat.synchronize(device)

        assert torch.equal(emb_i.weight.data[12:], orig_weight[12:])
        assert torch.equal(ema_w[12:], orig_ema_w[12:])
        assert torch.equal(ema_cs[12:], orig_ema_cs[12:])

        assert torch.isfinite(emb_i.weight.data[:12]).all()
        assert torch.isfinite(ema_w[:12]).all()
        assert torch.equal(emb_i.weight.data[:12], ema_w[:12].to(emb_i.weight.dtype))
        assert torch.equal(ema_cs[:12], torch.full((12,), 0.5, device=device, dtype=ema_cs.dtype))

    @pytest.mark.parametrize("device_str", ["cpu", "xpu:0"])
    def test_sample_codes_and_sample_level_raw(self, device_str):
        device = _resolve_test_device(device_str)
        torch.manual_seed(789)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=32,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[32, 32, 32],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).to(device).eval()

        vqvae.vq.set_stochastic_encode(True, temperature=0.7, top_k=8, top_p=0.9)
        distances = torch.rand(6, 32, device=device) * 5.0
        sampled_codes = vqvae.vq._sample_codes(distances)
        z = torch.randn(2, 4, 16, device=device)
        idx = vqvae.vq.z_to_idx(z)
        xpu_compat.synchronize(device)

        assert sampled_codes.shape == (6,)
        assert sampled_codes.dtype == torch.long
        assert ((sampled_codes >= 0) & (sampled_codes < 32)).all()
        assert idx.shape == (2, 7)

        model_cfg = OmegaConf.create({
            "n_layer": 2,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "context_length": 8,
            "prefix_len": 2,
            "raw_token_prefix": True,
            "prefix_pad_token_id": 0,
            "vocab_size": [32, 32, 32],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).to(device).eval()
        logits = torch.randn(2, 4, 32, device=device)
        sampled = net._sample_level_raw(logits, temperature=0.8, top_k=8, top_p=0.9)
        sampled_greedy = net._sample_level_raw(logits, temperature=0.0, top_k=8, top_p=0.9)
        xpu_compat.synchronize(device)

        assert sampled.shape == (2, 4)
        assert sampled.dtype == torch.long
        assert ((sampled >= 0) & (sampled < 32)).all()
        assert sampled_greedy.shape == (2, 4)
        assert torch.equal(sampled_greedy, logits.argmax(dim=-1))

    @pytest.mark.parametrize("device_str", ["cpu", "xpu:0"])
    @pytest.mark.parametrize("use_rope", [True, False])
    def test_dynamic_prefix_trim_equivalence_rope_and_wpe(self, device_str, use_rope):
        device = _resolve_test_device(device_str)
        torch.manual_seed(2026)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=32,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[8, 8, 8],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).to(device).eval()

        P = 64
        p_eff = 24
        model_cfg = OmegaConf.create({
            "n_layer": 2,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "context_length": 8,
            "prefix_len": P,
            "raw_token_prefix": True,
            "prefix_pad_token_id": 0,
            "vocab_size": [8, 8, 8],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": use_rope,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).to(device).eval()

        B = 2
        t_target = sum(net.levels[net._lvl_first:])
        x_full = torch.randn(B, P + t_target, vqvae.embed_dim, device=device)
        pad_mask_full = torch.zeros(B, P, dtype=torch.bool, device=device)
        pad_mask_full[:, 20:] = True

        x_trimmed = torch.cat([x_full[:, :p_eff, :], x_full[:, P:, :]], dim=1)
        pad_mask_trimmed = pad_mask_full[:, :p_eff]

        with torch.no_grad():
            logits_full = net(x_full, prefix_pad_mask=pad_mask_full)
            logits_trimmed = net(x_trimmed, prefix_pad_mask=pad_mask_trimmed)
        xpu_compat.synchronize(device)

        assert logits_trimmed.shape == logits_full.shape
        assert torch.isfinite(logits_trimmed).all()
        assert torch.allclose(logits_trimmed, logits_full, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("device_str", ["cpu", "xpu:0"])
    def test_all_pad_prefix_produces_finite_logits_and_grads(self, device_str):
        device = _resolve_test_device(device_str)
        torch.manual_seed(2027)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=32,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[8, 8, 8],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).to(device).eval()

        P = 16
        p_eff = 12
        model_cfg = OmegaConf.create({
            "n_layer": 2,
            "n_head": 2,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 32,
            "context_length": 8,
            "prefix_len": P,
            "raw_token_prefix": True,
            "prefix_pad_token_id": 0,
            "vocab_size": [8, 8, 8],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).to(device).train()

        B = 2
        t_target = sum(net.levels[net._lvl_first:])
        x = torch.randn(B, p_eff + t_target, vqvae.embed_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        all_pad_mask = torch.ones(B, p_eff, dtype=torch.bool, device=device)

        net.zero_grad(set_to_none=True)
        with xpu_compat.autocast_context(device_str, dtype=torch.bfloat16, enabled=True):
            logits = net(x, prefix_pad_mask=all_pad_mask)
            loss = logits.sum()
        loss.backward()
        xpu_compat.synchronize(device)

        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf for all-pad prefix"
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grad_checked = 0
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad_checked += 1
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"
        assert grad_checked > 0


class TestWorkerRecycleExit:
    @staticmethod
    def _setup_recycle_fixture(base_dir: Path) -> tuple[Path, Path, Path]:
        prefix_len = 16
        target_len = 8
        levels = [4, 8]
        concept_len = sum(levels)
        vocab_n = 32
        in_vocab = 64

        cfg = OmegaConf.create(
            {
                "task": "ncm",
                "model": {
                    "n_layer": 2,
                    "n_head": 2,
                    "bias": True,
                    "dropout": 0.0,
                    "n_embd": 32,
                    "context_length": prefix_len + concept_len,
                    "prefix_len": prefix_len,
                    "raw_token_prefix": True,
                    "prefix_pad_token_id": 63,
                    "vocab_size": [vocab_n] * len(levels),
                    "predict_level0": False,
                    "attn_pattern": "block_diagonal",
                    "use_positional_encoding": True,
                    "use_level_encoding": True,
                    "use_rope": True,
                    "rope_base": 100000,
                    "use_qk_norm": True,
                    "ctx_encoder": {"enabled": False, "n_layer": 1, "vocab_size": in_vocab},
                    "use_flex_attention": False,
                    "query_level0": True,
                    "tie_heads_to_codebook": True,
                    "tie_heads_learn_scale": True,
                },
                "tokenizer": {
                    "pre_tokenizer": {
                        "name": "hf",
                        "tokenizer": "hf",
                        "model_id": "gpt2",
                        "special_tokens": {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"},
                    },
                    "vqvae": {
                        "n_layers": 2,
                        "embed_dim": 16,
                        "in_vocab_size": in_vocab,
                        "context_length": target_len,
                        "compression_factor": 1,
                        "dropout": 0.0,
                        "pre_quant_groupnorm": 4,
                        "use_qk_norm": True,
                        "pre_quant_dropout": 0.0,
                        "enc_layers": 1,
                        "dec_layers": 1,
                        "vqvae_vocab_size": [vocab_n] * len(levels),
                        "checkpoint_path": "tokenizer.pt",
                        "vector_quantizer_config": {
                            "clss": "multiscale_residual_vector_quantizer",
                            "decay": 0.99,
                            "epsilon": 1.0e-5,
                            "commitment_cost": 0.25,
                            "restart_threshold": 1.0,
                            "kmeans_init": False,
                            "learned_l1_sampling": True,
                            "learned_l1_cascade": True,
                            "learned_l1_cascade_base": 2,
                            "learned_all_sampling": False,
                            "quant_resi": {
                                "enabled": True,
                                "ratio": 0.5,
                                "share_mode": 0,
                                "learnable_ratio": False,
                            },
                            "aux": {"fine_drop": {"prob": 0.5, "min_keep": 1}},
                            "levels": {"use_manual_levels": True, "manual_levels": levels},
                            "learned_l1_cascade_norm": True,
                            "learned_l1_cascade_norm_final": True,
                        },
                    },
                },
                "train": {
                    "stage2": {
                        "optimizer": "adamw",
                        "batch_size": 2,
                        "grad_accum": 1,
                        "lr": 1.0e-3,
                        "min_lr": 1.0e-4,
                        "warmup_steps": 1,
                        "max_steps": 6,
                        "stable_frac": 0.5,
                        "gradient_checkpointing": False,
                        "foreach_adamw": False,
                        "save_every": 2,
                        "eval_every": 6,
                        "log_every": 1,
                        "export_every": 2,
                    }
                },
            }
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = base_dir / "mini.yaml"
        OmegaConf.save(cfg, cfg_path)

        memmap_root = base_dir / "memmap"
        rng = np.random.default_rng(11)
        n_samples = 6
        for split_name in ("stage2_wide", "stage2_gold", "eval"):
            sd = memmap_root / split_name
            sd.mkdir(parents=True, exist_ok=True)
            np.asarray(rng.integers(0, 60, size=(n_samples, prefix_len)), dtype=np.uint16).tofile(
                sd / f"prefix_{prefix_len}.bin"
            )
            np.asarray(rng.integers(0, 60, size=(n_samples, target_len)), dtype=np.uint16).tofile(
                sd / f"target_{target_len}.bin"
            )
            np.asarray([i % 4 for i in range(n_samples)], dtype=np.uint8).tofile(sd / "modes.bin")
            (sd / "meta.json").write_text(
                json.dumps(
                    {
                        "num_samples": n_samples,
                        "prefix_len": prefix_len,
                        "target_len": target_len,
                        "concept_len": concept_len,
                        "dtype": "uint16",
                    }
                ),
                encoding="utf-8",
            )

        run_dir = base_dir / "run"
        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        vq_cfg = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True))
        vq = VQVAE(**vq_cfg)
        torch.save({"model": vq.state_dict()}, export_dir / "tokenizer.pt")
        return cfg_path, memmap_root, run_dir

    def test_train_ncm_recycle_every_exits_cleanly(self, tmp_path: Path, monkeypatch):
        from scripts import train_ncm

        cfg_path, memmap_root, run_dir = self._setup_recycle_fixture(tmp_path)

        orig_atomic_save = train_ncm._atomic_save
        latest_save_count = 0

        def _counting_atomic_save(obj, path: Path) -> None:
            nonlocal latest_save_count
            if Path(path) == run_dir / "ncm_latest.pt":
                latest_save_count += 1
            orig_atomic_save(obj, path)

        monkeypatch.setattr(train_ncm, "_atomic_save", _counting_atomic_save)

        rc = train_ncm.main(
            [
                "--config",
                str(cfg_path),
                "--memmap-root",
                str(memmap_root),
                "--run-dir",
                str(run_dir),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--recycle-every",
                "2",
                "--allow-recycle-exit",
            ]
        )

        assert rc == 0
        assert latest_save_count == 1
        assert (run_dir / "ncm_latest.pt").is_file()
        latest = torch.load(run_dir / "ncm_latest.pt", map_location="cpu", weights_only=False)
        assert latest["step"] == 2
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        assert status["step"] == 2
        assert status["state"] == "recycling"

    def test_train_ncm_commit_growth_vs_constant_high_baseline(self, tmp_path: Path, monkeypatch):
        from scripts import train_ncm

        cfg_path_a, memmap_root_a, run_dir_a = self._setup_recycle_fixture(tmp_path / "part_a")
        monkeypatch.setattr(xpu_compat, "get_process_commit_gb", lambda: 28.5)

        rc_a = train_ncm.main(
            [
                "--config",
                str(cfg_path_a),
                "--memmap-root",
                str(memmap_root_a),
                "--run-dir",
                str(run_dir_a),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "26",
                "--max-process-commit-gb",
                "45.0",
                "--max-commit-growth-gb",
                "6.0",
                "--allow-recycle-exit",
            ]
        )

        assert rc_a == 0
        status_a = json.loads((run_dir_a / "status.json").read_text(encoding="utf-8"))
        assert status_a["step"] == 26
        assert status_a["state"] == "completed"

        cfg_path_b, memmap_root_b, run_dir_b = self._setup_recycle_fixture(tmp_path / "part_b")
        current_step = {"value": 0}
        orig_wsd_schedule = train_ncm.wsd_schedule

        def _tracking_wsd_schedule(step: int, *args, **kwargs):
            current_step["value"] = step + 1
            return orig_wsd_schedule(step, *args, **kwargs)

        def _stateful_commit_gb() -> float:
            return 28.5 if current_step["value"] < 25 else 35.0

        monkeypatch.setattr(train_ncm, "wsd_schedule", _tracking_wsd_schedule)
        monkeypatch.setattr(xpu_compat, "get_process_commit_gb", _stateful_commit_gb)

        rc_b = train_ncm.main(
            [
                "--config",
                str(cfg_path_b),
                "--memmap-root",
                str(memmap_root_b),
                "--run-dir",
                str(run_dir_b),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-steps",
                "30",
                "--max-process-commit-gb",
                "45.0",
                "--max-commit-growth-gb",
                "6.0",
                "--allow-recycle-exit",
            ]
        )

        assert rc_b == 0
        status_b = json.loads((run_dir_b / "status.json").read_text(encoding="utf-8"))
        assert status_b["state"] == "recycling"
        assert status_b["step"] == 25


class TestHybridOptimizerAndAdaptiveCheckpointing:
    @staticmethod
    def _make_mixed_params(seed: int = 101) -> list[torch.nn.Parameter]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        shapes = [(8, 8), (4, 6), (12,), (3, 5), (7,)]
        params: list[torch.nn.Parameter] = []
        for shape in shapes:
            t = (torch.randn(shape, generator=g, dtype=torch.float32) * 0.25).to(torch.bfloat16)
            params.append(torch.nn.Parameter(t))
        return params

    @staticmethod
    def _assign_accum_grads(
        params_a: list[torch.nn.Parameter],
        params_b: list[torch.nn.Parameter],
        step_idx: int,
        micro_steps: int = 2,
    ) -> None:
        g = torch.Generator(device="cpu").manual_seed(5000 + step_idx)
        for pa, pb in zip(params_a, params_b):
            pa.grad = None
            pb.grad = None
        for _ in range(micro_steps):
            for pa, pb in zip(params_a, params_b):
                micro = (torch.randn(pa.shape, generator=g, dtype=torch.float32) * (0.1 / micro_steps)).to(
                    torch.bfloat16
                )
                pa.grad = micro.clone() if pa.grad is None else (pa.grad + micro)
                pb.grad = micro.clone() if pb.grad is None else (pb.grad + micro)

    def test_kahan_bf16_cpu_offload_bit_exact_and_multi_slab(self, monkeypatch):
        from scripts.train_ncm import KahanBF16AdamW

        monkeypatch.setattr(KahanBF16AdamW, "SLAB_MAX_ELEMS", 32)
        params_ref = self._make_mixed_params(seed=77)
        params_off = self._make_mixed_params(seed=77)

        opt_ref = KahanBF16AdamW(params_ref, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=False)
        opt_off = KahanBF16AdamW(params_off, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=True)

        for step_idx in range(5):
            self._assign_accum_grads(params_ref, params_off, step_idx=step_idx, micro_steps=2)
            opt_ref.step()
            opt_off.step()

        for p_ref, p_off in zip(params_ref, params_off):
            assert torch.equal(p_ref.data, p_off.data)
            st_ref = opt_ref.state[p_ref]
            st_off = opt_off.state[p_off]
            assert st_ref["step"] == st_off["step"] == 5
            expected_keys = (
                {"step", "kahan_comp", "exp_avg", "exp_avg_sq_row", "exp_avg_sq_col"}
                if p_ref.ndim == 2
                else {"step", "kahan_comp", "exp_avg", "exp_avg_sq"}
            )
            assert set(st_off.keys()) == expected_keys
            for key in expected_keys - {"step"}:
                assert st_off[key].device.type == "cpu"
                assert torch.equal(st_ref[key], st_off[key])

        sd_ref = opt_ref.state_dict()
        sd_off = opt_off.state_dict()
        assert set(sd_off.keys()) == set(sd_ref.keys()) == {"state", "param_groups"}
        assert set(sd_off["state"].keys()) == set(sd_ref["state"].keys())
        for pid in sd_ref["state"]:
            assert set(sd_off["state"][pid].keys()) == set(sd_ref["state"][pid].keys())

    def test_kahan_bf16_sync_shadows_and_cross_load_state_dict(self):
        import copy
        from scripts.train_ncm import KahanBF16AdamW

        params_ref = self._make_mixed_params(seed=202)
        params_mix = self._make_mixed_params(seed=202)

        opt_ref = KahanBF16AdamW(params_ref, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=False)
        opt_a = KahanBF16AdamW(params_mix, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=False)

        for step_idx in range(2):
            self._assign_accum_grads(params_ref, params_mix, step_idx=step_idx, micro_steps=2)
            opt_ref.step()
            opt_a.step()

        saved_sd_1 = copy.deepcopy(opt_a.state_dict())
        saved_weights_1 = [p.detach().clone() for p in params_mix]

        opt_b = KahanBF16AdamW(params_mix, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=True)
        with torch.no_grad():
            for p, w in zip(params_mix, saved_weights_1):
                p.copy_(w)
        opt_b.sync_cpu_shadows()
        opt_b.load_state_dict(saved_sd_1)
        for p in params_mix:
            for val in opt_b.state[p].values():
                if isinstance(val, torch.Tensor):
                    assert val.device.type == "cpu"

        for step_idx in range(2, 4):
            self._assign_accum_grads(params_ref, params_mix, step_idx=step_idx, micro_steps=2)
            opt_ref.step()
            opt_b.step()

        for p_ref, p_mix in zip(params_ref, params_mix):
            assert torch.equal(p_ref.data, p_mix.data)

        saved_sd_2 = copy.deepcopy(opt_b.state_dict())
        opt_c = KahanBF16AdamW(params_mix, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, cpu_offload=False)
        opt_c.load_state_dict(saved_sd_2)

        self._assign_accum_grads(params_ref, params_mix, step_idx=4, micro_steps=2)
        opt_ref.step()
        opt_c.step()

        for p_ref, p_mix in zip(params_ref, params_mix):
            assert torch.equal(p_ref.data, p_mix.data)
            for key in opt_ref.state[p_ref]:
                v_ref = opt_ref.state[p_ref][key]
                v_mix = opt_c.state[p_mix][key]
                if isinstance(v_ref, torch.Tensor):
                    assert torch.equal(v_ref, v_mix)
                else:
                    assert v_ref == v_mix

        with torch.no_grad():
            for p in params_mix:
                p.add_(1.0)
        new_weights_snapshot = [p.detach().clone() for p in params_mix]
        opt_b_fresh = KahanBF16AdamW(params_mix, lr=2e-4, cpu_offload=True)
        for p in params_mix:
            p.grad = torch.zeros_like(p)
        opt_b_fresh.step()
        assert len(opt_b_fresh._cpu_param_shadow) == len(params_mix)
        with torch.no_grad():
            for p, snap in zip(params_mix, new_weights_snapshot):
                p.copy_(snap + 0.5)
        opt_b_fresh.sync_cpu_shadows()
        assert opt_b_fresh._cpu_param_shadow == {}
        for p in params_mix:
            p.grad = torch.zeros_like(p)
        opt_b_fresh.step()
        for p, snap in zip(params_mix, new_weights_snapshot):
            assert torch.equal(p.data, (snap + 0.5).to(p.dtype))

    def test_length_adaptive_gradient_checkpointing_caching_and_grad_parity(self):
        from scripts.train_ncm import _enable_gradient_checkpointing

        torch.manual_seed(303)
        vq_cfg = OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.95,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "learned_l1_sampling": False,
            "learned_all_sampling": False,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4]},
        })
        vqvae = VQVAE(
            in_vocab_size=32,
            context_length=4,
            embed_dim=16,
            vqvae_vocab_size=[8, 8, 8],
            n_layers=2,
            vector_quantizer_config=vq_cfg,
            compression_factor=1,
            dropout=0.0,
            use_qk_norm=True,
        ).eval()

        model_cfg = OmegaConf.create({
            "n_layer": 4,
            "n_head": 4,
            "bias": True,
            "dropout": 0.0,
            "n_embd": 64,
            "context_length": 260,
            "prefix_len": 256,
            "raw_token_prefix": True,
            "prefix_pad_token_id": 0,
            "vocab_size": [8, 8, 8],
            "predict_level0": False,
            "attn_pattern": "block_diagonal",
            "use_positional_encoding": True,
            "use_level_encoding": True,
            "use_rope": True,
            "rope_base": 10000,
            "use_qk_norm": True,
            "use_flex_attention": False,
            "query_level0": True,
            "tie_heads_to_codebook": True,
            "tie_heads_learn_scale": True,
        })
        net = NextConceptTransformer(vqvae_local=vqvae, config=model_cfg).train()
        call_counts = [0] * len(net.transformer.blocks)
        for idx, block in enumerate(net.transformer.blocks):
            raw_fwd = block.forward

            def _make_spy(fn, b_idx: int):
                def _spy(x, *args, **kwargs):
                    call_counts[b_idx] += 1
                    return fn(x, *args, **kwargs)

                return _spy

            block.forward = _make_spy(raw_fwd, idx)

        _enable_gradient_checkpointing(net, activation_budget_mb=0.5)

        B = 2
        x_short = torch.randn(B, 25 + 7, 16)
        net.zero_grad(set_to_none=True)
        loss_short = net(x_short).sum()
        loss_short.backward()
        assert call_counts == [1, 1, 2, 2]
        grads_adaptive = [
            p.grad.detach().clone() if p.grad is not None else None for p in net.parameters()
        ]

        call_counts[:] = [0] * len(net.transformer.blocks)
        x_long = torch.randn(B, 249 + 7, 16)
        net.zero_grad(set_to_none=True)
        loss_long = net(x_long).sum()
        loss_long.backward()
        assert call_counts == [2, 2, 2, 2]

        _enable_gradient_checkpointing(net, activation_budget_mb=0.0)
        call_counts[:] = [0] * len(net.transformer.blocks)
        net.zero_grad(set_to_none=True)
        loss_ref = net(x_short).sum()
        loss_ref.backward()
        assert call_counts == [2, 2, 2, 2]

        for p, g_adapt in zip(net.parameters(), grads_adaptive):
            if g_adapt is None:
                assert p.grad is None
            else:
                assert p.grad is not None
                assert torch.allclose(p.grad, g_adapt, atol=1e-5, rtol=1e-5)

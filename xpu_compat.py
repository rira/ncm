from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import sys
import time
import urllib.request
from typing import Any

import numpy as np
import torch


def _coerce_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _mps_available() -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def is_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except Exception:
        return False


def get_discrete_xpu_indices() -> list[int]:
    if not is_xpu_available():
        return []
    try:
        count = int(torch.xpu.device_count())
    except Exception:
        return []
    if count <= 0:
        return []

    named: list[tuple[int, str]] = []
    for idx in range(count):
        try:
            name = str(torch.xpu.get_device_name(idx)).lower()
        except Exception:
            name = ""
        named.append((idx, name))

    arc_indices = [idx for idx, name in named if "arc" in name and "uhd" not in name]
    if arc_indices:
        return arc_indices
    non_uhd_indices = [idx for idx, name in named if "uhd" not in name]
    if non_uhd_indices:
        return non_uhd_indices
    return [idx for idx, _ in named]


def select_best_xpu_index() -> int:
    indices = get_discrete_xpu_indices()
    return indices[0] if indices else 0


def resolve_device(preferred: str | None = None) -> torch.device:
    pref = str(preferred).strip() if preferred is not None else ""
    if not pref or pref.lower() == "auto":
        pref = os.environ.get("NCM_DEVICE", "").strip()

    if not pref or pref.lower() == "auto":
        if is_xpu_available():
            dev = torch.device(f"xpu:{select_best_xpu_index()}")
        elif torch.cuda.is_available():
            dev = torch.device("cuda:0")
        elif _mps_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        low = pref.lower()
        if low == "xpu":
            idx = select_best_xpu_index() if is_xpu_available() else 0
            dev = torch.device(f"xpu:{idx}")
        elif low == "cuda":
            dev = torch.device("cuda:0")
        else:
            dev = torch.device(low)

    if dev.type == "xpu" and is_xpu_available():
        torch.xpu.set_device(dev)
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(dev)
    return dev


def _detect_backend() -> str:
    env_dev = os.environ.get("NCM_DEVICE", "").strip().lower()
    if env_dev.startswith("xpu"):
        return "xpu"
    if env_dev.startswith("cuda"):
        return "cuda"
    if env_dev == "mps":
        return "mps"
    if env_dev == "cpu":
        return "cpu"
    if is_xpu_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def get_worker_device_ids(gpus_arg: str | None = None) -> tuple[str, list[str]]:
    if gpus_arg is not None and gpus_arg.strip():
        tokens = [t.strip() for t in gpus_arg.split(",") if t.strip()]
        low_tokens = [t.lower() for t in tokens]
        if low_tokens == ["cpu"]:
            return ("cpu", ["cpu"])
        if low_tokens == ["mps"]:
            return ("mps", ["0"])
        if all(t.startswith("xpu:") for t in low_tokens):
            return ("xpu", [t.split(":", 1)[1] for t in low_tokens])
        if all(t.startswith("cuda:") for t in low_tokens):
            return ("cuda", [t.split(":", 1)[1] for t in low_tokens])
        return (_detect_backend(), tokens)

    env_dev = os.environ.get("NCM_DEVICE", "").strip().lower()
    if env_dev == "cpu":
        return ("cpu", ["cpu"])
    if env_dev == "mps":
        return ("mps", ["0"])
    if env_dev.startswith("xpu:"):
        return ("xpu", [env_dev.split(":", 1)[1]])
    if env_dev == "xpu" or (not env_dev and is_xpu_available()):
        return ("xpu", [str(i) for i in get_discrete_xpu_indices()])
    if env_dev.startswith("cuda:"):
        return ("cuda", [env_dev.split(":", 1)[1]])
    if env_dev == "cuda" or (not env_dev and torch.cuda.is_available()):
        return ("cuda", [str(i) for i in range(int(torch.cuda.device_count()))])
    if not env_dev and _mps_available():
        return ("mps", ["0"])
    return ("cpu", [])


def configure_worker_env(
    base_env: dict[str, str], backend_type: str, device_id: str
) -> dict[str, str]:
    env = dict(base_env)
    backend = backend_type.strip().lower()
    if backend == "xpu":
        env["NCM_DEVICE"] = device_id if device_id.startswith("xpu:") else f"xpu:{device_id}"
    elif backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        env["NCM_DEVICE"] = "cuda:0"
    elif backend == "mps":
        env["NCM_DEVICE"] = "mps"
    else:
        env["NCM_DEVICE"] = "cpu"
    return env


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        if is_xpu_available():
            torch.xpu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        elif _mps_available() and hasattr(torch, "mps"):
            torch.mps.synchronize()
        return

    dev = _coerce_device(device)
    if dev.type == "xpu" and is_xpu_available():
        if dev.index is not None:
            torch.xpu.synchronize(dev)
        else:
            torch.xpu.synchronize()
    elif dev.type == "cuda" and torch.cuda.is_available():
        if dev.index is not None:
            torch.cuda.synchronize(dev)
        else:
            torch.cuda.synchronize()
    elif dev.type == "mps" and _mps_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_cache(device: torch.device | str | None = None) -> None:
    if device is None:
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif _mps_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        return

    dev = _coerce_device(device)
    if dev.type == "xpu" and is_xpu_available():
        torch.xpu.empty_cache()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev.type == "mps" and _mps_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def shutdown_device_cleanly(device: torch.device | str | None = None) -> None:
    gc.collect()
    try:
        synchronize(device)
    except Exception:
        pass
    try:
        empty_cache(device)
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def supports_fp64(device: torch.device | str | None = None) -> bool:
    if device is None:
        if is_xpu_available():
            dev = torch.device(f"xpu:{select_best_xpu_index()}")
        elif torch.cuda.is_available():
            return True
        elif _mps_available():
            return False
        else:
            return True
    else:
        dev = _coerce_device(device)

    if dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not hasattr(xpu, "get_device_capability"):
            return False
        try:
            cap = xpu.get_device_capability(dev)
        except Exception:
            return False
        if isinstance(cap, dict):
            return bool(cap.get("has_fp64", False))
        return bool(getattr(cap, "has_fp64", False))
    if dev.type in ("cuda", "cpu"):
        return True
    if dev.type == "mps":
        return False
    return False


def autocast_context(
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    enabled: bool = True,
):
    if not enabled:
        return contextlib.nullcontext()
    dev = _coerce_device(device)
    dev_type = dev.type
    if dev_type == "xpu" and not is_xpu_available():
        return contextlib.nullcontext()
    if dev_type == "cuda" and not torch.cuda.is_available():
        return contextlib.nullcontext()
    if dev_type == "mps" and not _mps_available():
        return contextlib.nullcontext()

    is_avail = getattr(getattr(torch, "amp", None), "is_autocast_available", None)
    if callable(is_avail) and not is_avail(dev_type):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=dev_type, dtype=dtype, enabled=True)


def configure_matmul_precision(device: torch.device | str) -> None:
    dev = _coerce_device(device)
    if dev.type not in ("cuda", "xpu"):
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if dev.type == "cuda":
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = True
        cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn_backend is not None:
            cudnn_backend.allow_tf32 = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available() and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_vram_backend(dev: torch.device):
    if dev.type == "xpu" and is_xpu_available():
        return torch.xpu
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda
    return None


def get_vram_stats_mb(device: torch.device | str) -> dict[str, float]:
    dev = _coerce_device(device)
    mb = 1024.0 * 1024.0
    zero = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "total_mb": 0.0,
    }
    backend = _resolve_vram_backend(dev)
    if backend is None:
        return zero

    try:
        allocated = float(backend.memory_allocated(dev)) / mb
        reserved = float(backend.memory_reserved(dev)) / mb
        max_allocated = float(backend.max_memory_allocated(dev)) / mb
    except Exception:
        return zero

    total = 0.0
    if hasattr(backend, "get_device_properties"):
        try:
            props = backend.get_device_properties(dev)
            total = float(getattr(props, "total_memory", 0)) / mb
        except Exception:
            total = 0.0

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "total_mb": total,
    }


def configure_vram_cap(
    device: torch.device | str,
    *,
    max_reserved_mb: float = 0.0,
    memory_fraction: float = 0.0,
) -> dict[str, Any]:
    dev = _coerce_device(device)
    cap_mb = float(max(0.0, max_reserved_mb))
    backend = _resolve_vram_backend(dev)
    if backend is None or memory_fraction <= 0.0:
        return {"applied": False, "fraction": 0.0, "max_reserved_mb": cap_mb}

    frac = min(0.98, max(0.1, float(memory_fraction)))
    setter = getattr(backend, "set_per_process_memory_fraction", None)
    if not callable(setter):
        setter = getattr(getattr(backend, "memory", None), "set_per_process_memory_fraction", None)
    if not callable(setter):
        return {"applied": False, "fraction": 0.0, "max_reserved_mb": cap_mb}

    try:
        try:
            setter(frac, dev)
        except TypeError:
            setter(frac, dev.index if dev.index is not None else 0)
    except Exception:
        return {"applied": False, "fraction": 0.0, "max_reserved_mb": cap_mb}

    return {"applied": True, "fraction": frac, "max_reserved_mb": cap_mb}


_LAST_VRAM_TRIM_TS: float = 0.0
MIN_FREEABLE_SLACK_MB: float = 512.0


def trim_vram_cache_if_needed(
    device: torch.device | str,
    max_reserved_mb: float,
    *,
    min_interval_sec: float = 0.0,
) -> bool:
    global _LAST_VRAM_TRIM_TS
    if max_reserved_mb <= 0.0:
        return False
    if min_interval_sec > 0.0:
        now = time.monotonic()
        if _LAST_VRAM_TRIM_TS > 0.0 and (now - _LAST_VRAM_TRIM_TS) < float(min_interval_sec):
            return False
    dev = _coerce_device(device)
    backend = _resolve_vram_backend(dev)
    if backend is None:
        return False
    try:
        reserved_mb = float(backend.memory_reserved(dev)) / (1024.0 * 1024.0)
    except Exception:
        return False
    try:
        allocated_mb = float(backend.memory_allocated(dev)) / (1024.0 * 1024.0)
    except Exception:
        allocated_mb = 0.0
    if reserved_mb > float(max_reserved_mb) and (reserved_mb - allocated_mb) >= MIN_FREEABLE_SLACK_MB:
        empty_cache(dev)
        _LAST_VRAM_TRIM_TS = time.monotonic()
        return True
    return False


def get_process_commit_gb() -> float:
    if os.name != "nt":
        return 0.0
    import ctypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        windll = getattr(ctypes, "windll", None)
        kernel32 = getattr(windll, "kernel32", None)
        if kernel32 is None:
            return 0.0
        get_cur = kernel32.GetCurrentProcess
        get_cur.argtypes = []
        get_cur.restype = ctypes.c_void_p

        get_mem = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if get_mem is None:
            psapi = getattr(windll, "psapi", None)
            get_mem = getattr(psapi, "GetProcessMemoryInfo", None) if psapi is not None else None
        if get_mem is None:
            return 0.0

        get_mem.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), ctypes.c_uint32]
        get_mem.restype = ctypes.c_int

        pmc = _PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(pmc)
        ok = get_mem(get_cur(), ctypes.byref(pmc), pmc.cb)
        if ok:
            return float(pmc.PrivateUsage or pmc.PagefileUsage) / (1024.0**3)
        return 0.0
    except Exception:
        return 0.0


def lower_process_priority_for_desktop() -> bool:
    if os.name != "nt":
        return False
    import ctypes

    try:
        windll = getattr(ctypes, "windll", None)
        kernel32 = getattr(windll, "kernel32", None)
        if kernel32 is None:
            return False
        get_cur = kernel32.GetCurrentProcess
        get_cur.argtypes = []
        get_cur.restype = ctypes.c_void_p
        set_prio = kernel32.SetPriorityClass
        set_prio.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        set_prio.restype = ctypes.c_int
        handle = get_cur()
        ok = set_prio(handle, 0x00004000)
        return bool(ok)
    except Exception:
        return False


def check_ollama_vram_notice(host: str = "127.0.0.1", port: int = 11434) -> str | None:
    url = f"http://{host}:{port}/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    models = data.get("models") if isinstance(data, dict) else None
    if not models:
        return None

    has_vram_field = any(isinstance(m, dict) and "size_vram" in m for m in models)
    if has_vram_field:
        active = [m for m in models if isinstance(m, dict) and int(m.get("size_vram") or 0) > 0]
        if not active:
            return None
        total_bytes = sum(int(m.get("size_vram") or 0) for m in active)
    else:
        active = [m for m in models if isinstance(m, dict)]
        if not active:
            return None
        total_bytes = sum(int(m.get("size") or 0) for m in active)

    names = [str(m.get("name") or m.get("model") or "unknown") for m in active]
    gb = total_bytes / (1024**3)
    return f"Ollama VRAM active ({gb:.2f} GiB): {', '.join(names)}"

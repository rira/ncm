from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = os.environ.get("NCM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE

MODE_NAMES: dict[int, str] = {}


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
    os.replace(tmp_path, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    try:
        if device.type == "xpu" and xpu_compat.is_xpu_available() and hasattr(torch.xpu, "get_rng_state_all"):
            state["xpu"] = torch.xpu.get_rng_state_all()
        elif device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if device.type == "xpu" and "xpu" in state and hasattr(torch.xpu, "set_rng_state_all"):
            torch.xpu.set_rng_state_all(state["xpu"])
        elif device.type == "cuda" and "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


def _resolve_dtype(dtype_arg: str, device: torch.device) -> tuple[torch.dtype, bool]:
    low = str(dtype_arg or "bfloat16").lower()
    if low == "auto":
        dtype = torch.bfloat16 if device.type in ("xpu", "cuda") else torch.float32
    elif low == "bfloat16":
        dtype = torch.bfloat16
    elif low == "float16":
        dtype = torch.float16
    elif low == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype {dtype_arg!r}")
    autocast_enabled = device.type in ("xpu", "cuda") and dtype in (torch.bfloat16, torch.float16)
    return dtype, autocast_enabled


class KahanBF16AdamW(torch.optim.Optimizer):
    """Memory-efficient BF16 AdamW with Kahan summation and factored 2D second moments."""

    SLAB_MAX_ELEMS: int = 64 * 1024 * 1024

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        cpu_offload: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.cpu_offload = bool(cpu_offload)
        self._cpu_param_shadow: dict[torch.nn.Parameter, torch.Tensor] = {}

    def sync_cpu_shadows(self) -> None:
        self._cpu_param_shadow.clear()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cpu_param_shadow.clear()
        super().load_state_dict(state_dict)
        saved_state = state_dict.get("state", {})
        saved_groups = state_dict.get("param_groups", [])
        id_map = dict(
            zip(
                [pid for g in saved_groups for pid in g.get("params", [])],
                [p for g in self.param_groups for p in g["params"]],
            )
        )
        for saved_id, p in id_map.items():
            if saved_id not in saved_state or p not in self.state:
                continue
            src = saved_state[saved_id]
            dst = self.state[p]
            target_dev = torch.device("cpu") if self.cpu_offload else p.device
            for fp32_key in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                if fp32_key in src and isinstance(src[fp32_key], torch.Tensor):
                    dst[fp32_key] = src[fp32_key].detach().to(dtype=torch.float32, device=target_dev).clone()
            for bf16_key in ("kahan_comp", "exp_avg"):
                if bf16_key in src and isinstance(src[bf16_key], torch.Tensor):
                    dst[bf16_key] = src[bf16_key].detach().to(dtype=torch.bfloat16, device=target_dev).clone()

    @staticmethod
    def _apply_kahan_adamw_update(
        param_buf: torch.Tensor,
        grad_tensor: torch.Tensor,
        state: dict[str, Any],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
        state_dev: torch.device,
    ) -> torch.Tensor:
        if len(state) == 0:
            state["step"] = 0
            state["kahan_comp"] = torch.zeros(param_buf.shape, dtype=torch.bfloat16, device=state_dev)
            state["exp_avg"] = torch.zeros(param_buf.shape, dtype=torch.bfloat16, device=state_dev)
            if param_buf.ndim == 2:
                state["exp_avg_sq_row"] = torch.zeros(param_buf.shape[0], dtype=torch.float32, device=state_dev)
                state["exp_avg_sq_col"] = torch.zeros(param_buf.shape[1], dtype=torch.float32, device=state_dev)
            else:
                state["exp_avg_sq"] = torch.zeros(param_buf.shape, dtype=torch.float32, device=state_dev)

        kahan_comp = state["kahan_comp"]
        if kahan_comp.device != state_dev or kahan_comp.dtype != torch.bfloat16:
            kahan_comp = kahan_comp.to(device=state_dev, dtype=torch.bfloat16)
            state["kahan_comp"] = kahan_comp

        exp_avg = state["exp_avg"]
        if exp_avg.device != state_dev or exp_avg.dtype != torch.bfloat16:
            exp_avg = exp_avg.to(device=state_dev, dtype=torch.bfloat16)
            state["exp_avg"] = exp_avg

        state["step"] = int(state["step"]) + 1
        t = state["step"]

        g = grad_tensor.float()

        m = exp_avg.float().mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg.copy_(m.to(torch.bfloat16))
        m_hat = m / (1.0 - beta1 ** t)

        if param_buf.ndim == 2:
            r = state["exp_avg_sq_row"]
            c = state["exp_avg_sq_col"]
            if r.device != state_dev or r.dtype != torch.float32:
                r = r.to(device=state_dev, dtype=torch.float32)
                state["exp_avg_sq_row"] = r
            if c.device != state_dev or c.dtype != torch.float32:
                c = c.to(device=state_dev, dtype=torch.float32)
                state["exp_avg_sq_col"] = c
            g_sq = g.square()
            r_upd = (g_sq.sum(dim=1) / float(g_sq.shape[1])).clamp_min(0.0)
            c_upd = (g_sq.sum(dim=0) / float(g_sq.shape[0])).clamp_min(0.0)
            r.mul_(beta2).add_(r_upd, alpha=1.0 - beta2)
            c.mul_(beta2).add_(c_upd, alpha=1.0 - beta2)
            r_hat = (r / (1.0 - beta2 ** t)).clamp_min(0.0)
            c_hat = (c / (1.0 - beta2 ** t)).clamp_min(0.0)
            r_mean = (r_hat.sum() / float(r_hat.numel())).clamp(min=1e-30)
            v_hat = ((r_hat[:, None] * c_hat[None, :]) / r_mean).clamp_min(0.0)
        else:
            v = state["exp_avg_sq"]
            if v.device != state_dev or v.dtype != torch.float32:
                v = v.to(device=state_dev, dtype=torch.float32)
                state["exp_avg_sq"] = v
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            v_hat = (v / (1.0 - beta2 ** t)).clamp_min(0.0)

        p_fp32 = param_buf.float() + kahan_comp.float()
        if wd > 0 and param_buf.ndim >= 2:
            p_fp32.mul_(1.0 - lr * wd)
        p_fp32.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        p_bf16 = p_fp32.to(torch.bfloat16)
        kahan_comp.copy_((p_fp32 - p_bf16.float()).to(torch.bfloat16))
        return p_bf16

    @torch.no_grad()
    def step(self, closure: Any = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.cpu_offload:
            for group in self.param_groups:
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                wd = float(group["weight_decay"])

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("KahanBF16AdamW does not support sparse gradients")
                    p_bf16 = self._apply_kahan_adamw_update(
                        p,
                        p.grad,
                        self.state[p],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        wd=wd,
                        state_dev=p.device,
                    )
                    p.copy_(p_bf16)
            return loss

        non_cpu_devs: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("KahanBF16AdamW does not support sparse gradients")
                if p.device.type != "cpu":
                    non_cpu_devs.add(p.device)
        for dev in non_cpu_devs:
            xpu_compat.synchronize(dev)

        cpu_dev = torch.device("cpu")
        slab_limit = max(1, int(self.SLAB_MAX_ELEMS))

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            by_device: dict[torch.device, list[torch.nn.Parameter]] = {}
            for p in group["params"]:
                if p.grad is not None:
                    by_device.setdefault(p.device, []).append(p)

            for dev, dev_params in by_device.items():
                slabs: list[list[torch.nn.Parameter]] = []
                cur_slab: list[torch.nn.Parameter] = []
                cur_elems = 0
                for p in dev_params:
                    n_el = p.numel()
                    if cur_slab and (cur_elems + n_el > slab_limit):
                        slabs.append(cur_slab)
                        cur_slab = [p]
                        cur_elems = n_el
                    else:
                        cur_slab.append(p)
                        cur_elems += n_el
                if cur_slab:
                    slabs.append(cur_slab)

                for slab in slabs:
                    for p in slab:
                        if p not in self._cpu_param_shadow:
                            self._cpu_param_shadow[p] = p.detach().to(device=cpu_dev, dtype=p.dtype).clone()

                    cpu_grad_slab: torch.Tensor | None = None
                    if dev.type == "cpu":
                        cpu_grads = [p.grad.detach() for p in slab]
                    else:
                        grad_slab_dtype = (
                            torch.bfloat16
                            if all(p.grad.dtype == torch.bfloat16 for p in slab)
                            else torch.float32
                        )
                        if len(slab) == 1:
                            dev_grad_slab = slab[0].grad.detach().reshape(-1).to(grad_slab_dtype)
                        else:
                            dev_grad_slab = torch.cat(
                                [p.grad.detach().reshape(-1).to(grad_slab_dtype) for p in slab],
                                dim=0,
                            )
                        cpu_grad_slab = dev_grad_slab.to(device=cpu_dev, non_blocking=False)
                        del dev_grad_slab
                        cpu_grads = []
                        offset = 0
                        for p in slab:
                            n_el = p.numel()
                            cpu_grads.append(cpu_grad_slab[offset : offset + n_el].view_as(p))
                            offset += n_el

                    for p, g_cpu in zip(slab, cpu_grads):
                        shadow = self._cpu_param_shadow[p]
                        p_bf16 = self._apply_kahan_adamw_update(
                            shadow,
                            g_cpu,
                            self.state[p],
                            lr=lr,
                            beta1=beta1,
                            beta2=beta2,
                            eps=eps,
                            wd=wd,
                            state_dev=cpu_dev,
                        )
                        updated_val = p_bf16 if p.dtype == torch.bfloat16 else p_bf16.to(p.dtype)
                        shadow.copy_(updated_val)

                    del cpu_grads
                    if cpu_grad_slab is not None:
                        del cpu_grad_slab

                    if dev.type == "cpu":
                        for p in slab:
                            p.copy_(self._cpu_param_shadow[p])
                    else:
                        w_slab_dtype = (
                            torch.bfloat16
                            if all(p.dtype == torch.bfloat16 for p in slab)
                            else torch.float32
                        )
                        if len(slab) == 1:
                            cpu_w_slab = self._cpu_param_shadow[slab[0]].reshape(-1).to(w_slab_dtype)
                        else:
                            cpu_w_slab = torch.cat(
                                [self._cpu_param_shadow[p].reshape(-1).to(w_slab_dtype) for p in slab],
                                dim=0,
                            )
                        dev_w_slab = cpu_w_slab.to(device=dev, non_blocking=False)
                        del cpu_w_slab
                        offset = 0
                        for p in slab:
                            n_el = p.numel()
                            sl = dev_w_slab[offset : offset + n_el]
                            p.copy_(sl.view_as(p).to(p.dtype))
                            offset += n_el
                        del dev_w_slab

        return loss


class _MemmapStage2Dataset(Dataset):
    def __init__(self, split_dir: Path, prefix_len: int, target_len: int, concept_len: int):
        self.split_dir = Path(split_dir)
        meta_path = self.split_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.prefix_len = int(meta.get("prefix_len") or prefix_len)
        self.target_len = int(meta.get("target_len") or target_len)
        self.concept_len = int(meta.get("concept_len") or concept_len)
        self.np_dtype = np.dtype(meta.get("dtype") or "uint16")

        prefix_candidates = []
        if meta.get("prefix_file"):
            prefix_candidates.append(self.split_dir / str(meta["prefix_file"]))
        prefix_candidates.extend([
            self.split_dir / f"prefix_{self.prefix_len}.bin",
            self.split_dir / "prefix_1024.bin",
            self.split_dir / "prefix.bin",
        ])
        prefix_path = next((p for p in prefix_candidates if p.exists()), None)
        if prefix_path is None:
            raise FileNotFoundError(f"Missing prefix memmap binary under {self.split_dir}")

        target_candidates = []
        if meta.get("target_file"):
            target_candidates.append(self.split_dir / str(meta["target_file"]))
        target_candidates.extend([
            self.split_dir / f"target_{self.target_len}.bin",
            self.split_dir / "target_256.bin",
            self.split_dir / "target.bin",
        ])
        target_path = next((p for p in target_candidates if p.exists()), None)
        if target_path is None:
            raise FileNotFoundError(f"Missing target memmap binary under {self.split_dir}")

        inferred_n = prefix_path.stat().st_size // (self.prefix_len * self.np_dtype.itemsize)
        if "dtype" not in meta and prefix_path.stat().st_size % (self.prefix_len * 2) != 0 and prefix_path.stat().st_size % (self.prefix_len * 4) == 0:
            self.np_dtype = np.dtype("int32")
            inferred_n = prefix_path.stat().st_size // (self.prefix_len * 4)

        n_samples = int(
            meta.get("num_samples")
            or meta.get("num_rows")
            or meta.get("n_samples")
            or meta.get("count")
            or inferred_n
        )
        n_samples = min(n_samples, inferred_n) if inferred_n > 0 else 0
        if n_samples <= 0:
            raise ValueError(f"Empty stage2 dataset at {self.split_dir}")
        self.num_samples = n_samples

        self._prefix_mm = np.memmap(prefix_path, dtype=self.np_dtype, mode="r", shape=(self.num_samples, self.prefix_len))
        self._target_mm = np.memmap(target_path, dtype=self.np_dtype, mode="r", shape=(self.num_samples, self.target_len))

        gt_candidates = []
        if meta.get("gt_idx_file"):
            gt_candidates.append(self.split_dir / str(meta["gt_idx_file"]))
        gt_candidates.extend([
            self.split_dir / f"gt_idx_{self.concept_len}.bin",
            self.split_dir / "gt_idx.bin",
        ])
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        if gt_path is not None:
            gt_dtype = np.dtype(meta.get("gt_idx_dtype") or self.np_dtype)
            if gt_path.stat().st_size == self.num_samples * self.concept_len * 4:
                gt_dtype = np.dtype("int32")
            elif gt_path.stat().st_size == self.num_samples * self.concept_len * 2:
                gt_dtype = np.dtype("uint16")
            self._gt_mm = np.memmap(gt_path, dtype=gt_dtype, mode="r", shape=(self.num_samples, self.concept_len))
        else:
            self._gt_mm = None

        mode_candidates = []
        if meta.get("mode_id_file"):
            mode_candidates.append(self.split_dir / str(meta["mode_id_file"]))
        if meta.get("modes_file"):
            mode_candidates.append(self.split_dir / str(meta["modes_file"]))
        mode_candidates.extend([
            self.split_dir / "mode_id.bin",
            self.split_dir / "modes.bin",
            self.split_dir / "mode_ids.bin",
            self.split_dir / "mode.bin",
        ])
        mode_path = next((p for p in mode_candidates if p.exists()), None)
        if mode_path is not None:
            m_dtype = np.dtype("int32") if mode_path.stat().st_size == self.num_samples * 4 else np.dtype("uint8")
            self._mode_mm = np.memmap(mode_path, dtype=m_dtype, mode="r", shape=(self.num_samples,))
        else:
            self._mode_mm = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        prefix = torch.from_numpy(np.array(self._prefix_mm[idx], dtype=np.int64))
        target = torch.from_numpy(np.array(self._target_mm[idx], dtype=np.int64))
        gt_idx = torch.from_numpy(np.array(self._gt_mm[idx], dtype=np.int64)) if self._gt_mm is not None else None
        mode = int(self._mode_mm[idx]) if self._mode_mm is not None else 0
        return prefix, target, gt_idx, mode


def _collate_stage2(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]]):
    prefixes = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    has_gt = all(b[2] is not None for b in batch)
    gt_idx = torch.stack([b[2] for b in batch if b[2] is not None], dim=0) if has_gt else None
    modes = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return prefixes, targets, gt_idx, modes


def _enable_gradient_checkpointing(net: NextConceptTransformer, activation_budget_mb: float = 0.0) -> None:
    budget_val = max(0.0, float(activation_budget_mb))
    n_blocks = len(net.transformer.blocks)
    for block_idx, block in enumerate(net.transformer.blocks):
        block._ncm_activation_budget_mb = budget_val  # type: ignore[attr-defined]
        if getattr(block, "_ncm_grad_ckpt_wrapped", False):
            continue
        orig_forward = block.forward

        def make_ckpt_forward(fn, mod, b_idx: int, total_blocks: int):
            def ckpt_forward(x, *args, **kwargs):
                if mod.training and torch.is_grad_enabled():
                    cur_budget = float(getattr(mod, "_ncm_activation_budget_mb", 0.0))
                    if cur_budget > 0.0 and isinstance(x, torch.Tensor) and x.ndim == 3:
                        b_sz = max(1, int(x.shape[0]))
                        s_len = max(1, int(x.shape[1]))
                        n_embd = max(1, int(x.shape[2]))
                        n_head = int(getattr(getattr(mod, "attn", None), "n_head", 32))
                        est_block_mb = (
                            float(b_sz)
                            * (
                                float(n_head) * float(s_len) * float(s_len) * 4.2
                                + float(s_len) * float(n_embd) * 36.0
                            )
                            / (1024.0 * 1024.0)
                        )
                        cached_blocks = min(total_blocks, max(0, int(cur_budget // max(0.01, est_block_mb))))
                        if b_idx < cached_blocks:
                            return fn(x, *args, **kwargs)
                    return torch.utils.checkpoint.checkpoint(
                        lambda inp: fn(inp, *args, **kwargs),
                        x,
                        use_reentrant=False,
                    )
                return fn(x, *args, **kwargs)

            return ckpt_forward

        block.forward = make_ckpt_forward(orig_forward, block, block_idx, n_blocks)
        block._ncm_grad_ckpt_wrapped = True


def wsd_schedule(
    step: int,
    max_steps: int,
    warmup_steps: int,
    stable_frac: float,
    lr: float,
    min_lr: float,
    continuous: bool = False,
) -> tuple[str, float, str]:
    stable_end = max(warmup_steps, int(round(float(stable_frac) * max_steps)))
    stable_end = min(stable_end, max_steps)

    if step < max_steps:
        if warmup_steps > 0 and step < warmup_steps:
            cur_lr = lr * float(step + 1) / float(warmup_steps)
            return "warmup", cur_lr, "wide"
        if step < stable_end:
            return "stable", lr, "wide"
        decay_span = max(1, max_steps - stable_end)
        progress = min(max(float(step - stable_end) / float(decay_span), 0.0), 1.0)
        cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        return "decay", cur_lr, "gold"

    if not continuous:
        return "decay", min_lr, "gold"

    cycle_len = max(4, max_steps - stable_end)
    rel = step - max_steps
    cycle_idx = rel // cycle_len + 1
    c_pos = rel % cycle_len
    plateau_end = max(1, int(round(0.7 * cycle_len)))
    cycle_peak = min_lr + 0.5 * (lr - min_lr)
    if c_pos < plateau_end:
        return f"cycle_{cycle_idx}_stable", cycle_peak, "wide"
    anneal_span = max(1, cycle_len - plateau_end)
    progress = min(max(float(c_pos - plateau_end) / float(anneal_span), 0.0), 1.0)
    cur_lr = min_lr + 0.5 * (cycle_peak - min_lr) * (1.0 + math.cos(math.pi * progress))
    return f"cycle_{cycle_idx}_anneal", cur_lr, "gold"


def _load_or_build_unigrams(
    memmap_root: Path,
    in_vocab_size: int,
    l0_vocab_size: int,
    datasets: list[_MemmapStage2Dataset],
) -> tuple[torch.Tensor, torch.Tensor]:
    for candidate in (
        memmap_root / "unigrams.pt",
        memmap_root / "stage2_wide" / "unigrams.pt",
        memmap_root / "stage2_gold" / "unigrams.pt",
    ):
        if candidate.exists():
            obj = torch.load(candidate, map_location="cpu", weights_only=False)
            inf = obj.get("inference", obj) if isinstance(obj, dict) else {}
            if "token_unigram" in inf and "level0_unigram" in inf:
                tok_u = inf["token_unigram"].to(dtype=torch.float32).cpu()
                l0_u = inf["level0_unigram"].to(dtype=torch.float32).cpu()
                tok_u = tok_u / tok_u.sum().clamp(min=1e-12)
                l0_u = l0_u / l0_u.sum().clamp(min=1e-12)
                return tok_u, l0_u

    tok_counts = np.ones(in_vocab_size, dtype=np.float64)
    l0_counts = np.ones(l0_vocab_size, dtype=np.float64)
    for ds in datasets:
        sample_n = min(len(ds), 4096)
        if sample_n > 0:
            t_arr = np.asarray(ds._target_mm[:sample_n], dtype=np.int64).ravel()
            t_arr = t_arr[(t_arr >= 0) & (t_arr < in_vocab_size)]
            np.add.at(tok_counts, t_arr, 1.0)
            if ds._gt_mm is not None:
                l0_len = min(4, ds._gt_mm.shape[1])
                g_arr = np.asarray(ds._gt_mm[:sample_n, :l0_len], dtype=np.int64).ravel()
                g_arr = g_arr[(g_arr >= 0) & (g_arr < l0_vocab_size)]
                np.add.at(l0_counts, g_arr, 1.0)

    tok_u = torch.from_numpy((tok_counts / tok_counts.sum()).astype(np.float32))
    l0_u = torch.from_numpy((l0_counts / l0_counts.sum()).astype(np.float32))
    return tok_u, l0_u


def _export_eval_config(train_cfg_path: Path, cfg: Any, out_path: Path) -> None:
    stem = train_cfg_path.name
    candidate_eval = Path(ROOT) / "config" / "eval" / stem
    if candidate_eval.exists():
        eval_cfg = OmegaConf.load(candidate_eval)
    else:
        raw = OmegaConf.to_container(cfg, resolve=True) or {}
        raw.pop("train", None)
        eval_cfg = OmegaConf.create(
            {
                "task": raw.get("task", "ncm"),
                "checkpoints": {"generator": "generator.pt", "tokenizer": "tokenizer.pt"},
                "data": {"test": "eval_prompts.pickle", "test_refs": "eval_refs.pickle"},
                "model": raw["model"],
                "tokenizer": raw["tokenizer"],
                "generation": {
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 0.0,
                    "cfg_scale": 0.0,
                    "frequency_penalty": 2.0,
                    "frequency_deficit": 1.0,
                    "adjacency_penalty": "5,4,0.5",
                    "frequency_allowance": "generator.pt:inference.token_unigram",
                    "level0_unigram": "generator.pt:inference.level0_unigram",
                    "cut_at_eos": True,
                },
                "eval": {
                    "split": "test",
                    "n_docs": 1000,
                    "batch_size": 16,
                    "n_shards": 1,
                    "metric": "rouge",
                },
            }
        )
    eval_cfg.checkpoints.generator = "generator.pt"
    eval_cfg.checkpoints.tokenizer = "tokenizer.pt"
    eval_cfg.tokenizer.vqvae.checkpoint_path = "tokenizer.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    OmegaConf.save(eval_cfg, tmp_path)
    os.replace(tmp_path, out_path)


def _clip_grad_norm_fp32(params: list[torch.nn.Parameter], max_norm: float = 1.0) -> tuple[float, bool]:
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0, True
    total_norm_sq = torch.zeros((), dtype=torch.float32, device=grads[0].device)
    for g in grads:
        g_norm = torch.linalg.vector_norm(g.detach().float()).to(grads[0].device)
        total_norm_sq = total_norm_sq + g_norm * g_norm
    total_norm = torch.sqrt(total_norm_sq)
    if not torch.isfinite(total_norm):
        return float(total_norm.item()), False
    norm_val = float(total_norm.item())
    if max_norm > 0 and norm_val > max_norm:
        clip_coef = max_norm / (norm_val + 1e-6)
        for g in grads:
            g.detach().mul_(clip_coef)
    return norm_val, True


def _snapshot_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.device.type == "cpu" else obj.detach().to("cpu", non_blocking=False)
    if isinstance(obj, dict):
        return {k: _snapshot_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snapshot_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_snapshot_to_cpu(v) for v in obj)
    return obj


def _trim_prefix_to_bucket(
    prefix_ids: torch.Tensor,
    pad_token_id: int,
    bucket_size: int = 256,
    min_tokens: int = 4,
) -> torch.Tensor:
    if bucket_size <= 0 or prefix_ids.ndim != 2:
        return prefix_ids
    eff_bucket = max(int(bucket_size), int(min_tokens), 64)
    p_full = int(prefix_ids.shape[1])
    if p_full <= eff_bucket:
        return prefix_ids
    non_pad = prefix_ids != pad_token_id
    if bool(non_pad[:, -1].any().item()):
        return prefix_ids
    pos_idx = torch.arange(1, p_full + 1, device=prefix_ids.device, dtype=torch.long).unsqueeze(0)
    max_len = int(torch.where(non_pad, pos_idx, torch.zeros_like(pos_idx)).max().item())
    min_p = max(int(min_tokens), eff_bucket)
    p_eff = min(p_full, max(min_p, ((max_len + eff_bucket - 1) // eff_bucket) * eff_bucket))
    return prefix_ids[:, :p_eff].contiguous() if p_eff < p_full else prefix_ids


def _forward_ncm_step(
    net: NextConceptTransformer,
    vqvae: VQVAE,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    gt_idx: torch.Tensor | None,
    pad_token_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = prefix_ids.shape[0]
    if not getattr(net, "use_ctx_encoder", False) and not bool(getattr(net, "_cpu_prefix_trimmed", False)):
        bucket = max(int(getattr(net, "_prefix_bucket_size", 64)), 64)
        prefix_ids = _trim_prefix_to_bucket(prefix_ids, pad_token_id, bucket, int(net.levels[0]))
    net_dtype = net.transformer.wte.weight.dtype
    vq_dtype = vqvae.encoder.wte.weight.dtype
    step_autocast = autocast_enabled or (net_dtype == torch.bfloat16 or vq_dtype == torch.bfloat16)
    step_dtype = (
        torch.bfloat16
        if (not autocast_enabled and (net_dtype == torch.bfloat16 or vq_dtype == torch.bfloat16))
        else autocast_dtype
    )
    with xpu_compat.autocast_context(device, dtype=step_dtype, enabled=step_autocast):
        with torch.no_grad():
            if gt_idx is None:
                gt_idx = vqvae.encode(target_ids)
            z_prefix = vqvae.encoder.wte(prefix_ids)
            x_1L = vqvae.vq.idx_to_ncm_input(gt_idx)
        q0_dtype = z_prefix.dtype if net_dtype == torch.float32 else net_dtype
        q0 = net.build_l0_query_input(B, dtype=q0_dtype)
        x_in = torch.cat([z_prefix.to(q0.dtype), q0, x_1L.to(q0.dtype)], dim=1)
        prefix_pad_mask = prefix_ids == pad_token_id
        logits = net(x_in, prefix_pad_mask=prefix_pad_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), gt_idx.reshape(-1))
    return loss, logits, gt_idx


@torch.no_grad()
def _evaluate_ncm(
    net: NextConceptTransformer,
    vqvae: VQVAE,
    eval_loader: DataLoader,
    pad_token_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool,
    max_batches: int = 32,
    desktop_yield_ms: float = 0.0,
    mode_names: dict[int, str] | None = None,
) -> tuple[float, float, dict[str, float]]:
    was_training = net.training
    net.eval()
    total_loss = 0.0
    total_batches = 0
    l0_correct = 0
    l0_total = 0
    l0_len = int(vqvae.vq.levels[0])
    mode_loss_sum: dict[int, float] = {}
    mode_count: dict[int, int] = {}

    raw_bucket = int(getattr(net, "_prefix_bucket_size", 256))
    for b_idx, (prefix_ids, target_ids, gt_idx, modes) in enumerate(eval_loader):
        if b_idx >= max_batches:
            break
        if not getattr(net, "use_ctx_encoder", False) and raw_bucket > 0:
            prefix_ids = _trim_prefix_to_bucket(
                prefix_ids, pad_token_id, max(raw_bucket, 64), int(net.levels[0])
            )
        prefix_ids = prefix_ids.to(device, non_blocking=False)
        target_ids = target_ids.to(device, non_blocking=False)
        if gt_idx is not None:
            gt_idx = gt_idx.to(device, non_blocking=False)

        loss, logits, resolved_gt = _forward_ncm_step(
            net,
            vqvae,
            prefix_ids,
            target_ids,
            gt_idx,
            pad_token_id,
            device,
            autocast_dtype,
            autocast_enabled,
        )
        total_loss += float(loss.item())
        total_batches += 1

        pred_l0 = logits[:, :l0_len, :].argmax(dim=-1)
        true_l0 = resolved_gt[:, :l0_len]
        l0_correct += int((pred_l0 == true_l0).sum().item())
        l0_total += int(true_l0.numel())

        per_sample_loss = F.cross_entropy(
            logits.float().transpose(1, 2),
            resolved_gt,
            reduction="none",
        ).mean(dim=1)
        for m_val, s_loss in zip(modes.tolist(), per_sample_loss.tolist()):
            mode_loss_sum[m_val] = mode_loss_sum.get(m_val, 0.0) + float(s_loss)
            mode_count[m_val] = mode_count.get(m_val, 0) + 1
        if desktop_yield_ms > 0.0 and device.type in ("xpu", "cuda"):
            xpu_compat.synchronize(device)
            time.sleep(desktop_yield_ms / 1000.0)

    if was_training:
        net.train()

    if total_batches == 0:
        return float("inf"), 0.0, {}
    resolved_mode_names = mode_names if mode_names is not None else MODE_NAMES
    per_mode = {
        resolved_mode_names.get(m, f"MODE_{m}"): mode_loss_sum[m] / mode_count[m]
        for m in sorted(mode_count)
    }
    l0_acc = float(l0_correct) / float(max(1, l0_total))
    return total_loss / total_batches, l0_acc, per_mode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 2 Resumable WSD Next-Concept Trainer for Intel XPU / CUDA / CPU")
    ap.add_argument("--config", required=True, help="Path to config YAML (e.g. config/train/ncm_1_3b.yaml)")
    ap.add_argument("--memmap-root", default="data/memmap", help="Root directory containing stage2_wide/stage2_gold/eval")
    ap.add_argument("--tokenizer-ckpt", default=None, help="Path to VQ-VAE tokenizer checkpoint (default: <run-dir>/export/tokenizer.pt)")
    ap.add_argument("--run-dir", default="runs/ncm_1_3b", help="Run directory for checkpoints, status, and exports")
    ap.add_argument("--device", default="auto", help="Compute device (auto, xpu, cuda, cpu)")
    ap.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--max-steps", type=int, default=None, help="Override train.stage2.max_steps")
    ap.add_argument("--batch-size", type=int, default=None, help="Override train.stage2.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="Override train.stage2.grad_accum")
    ap.add_argument("--lr", type=float, default=None, help="Override train.stage2.lr")
    ap.add_argument("--min-lr", type=float, default=None, help="Override train.stage2.min_lr")
    ap.add_argument("--warmup-steps", type=int, default=None, help="Override train.stage2.warmup_steps")
    ap.add_argument("--continuous", action="store_true", help="Continue into plateau+anneal cycles after max_steps")
    ap.add_argument("--log-every", type=int, default=None, help="Override train.stage2.log_every")
    ap.add_argument("--eval-every", type=int, default=None, help="Override train.stage2.eval_every")
    ap.add_argument("--save-every", type=int, default=None, help="Override train.stage2.save_every")
    ap.add_argument("--export-every", type=int, default=None, help="Override train.stage2.export_every")
    ap.add_argument("--prefix-bucket-size", type=int, default=None, help="Override train.stage2.prefix_bucket_size")
    ap.add_argument("--max-reserved-vram-mb", type=float, default=None, help="Override train.stage2.max_reserved_vram_mb")
    ap.add_argument("--vram-trim-cooldown-sec", type=float, default=None, help="Minimum seconds between VRAM cache trims")
    ap.add_argument("--xpu-memory-fraction", type=float, default=None, help="Override train.stage2.xpu_memory_fraction")
    ap.add_argument("--desktop-yield-ms", type=float, default=None, help="Override train.stage2.desktop_yield_ms")
    ap.add_argument("--recycle-every", type=int, default=None, help="Cleanly exit after N steps when recycle exit is enabled")
    ap.add_argument("--max-process-commit-gb", type=float, default=None, help="Cleanly exit when Windows process commit exceeds GB")
    ap.add_argument("--max-commit-growth-gb", type=float, default=None, help="Cleanly exit when process commit grows by N GB above step-3 baseline")
    ap.add_argument("--allow-recycle-exit", action="store_true", help="Allow clean worker recycling exits")
    ap.add_argument("--no-below-normal-priority", action="store_true", help="Disable below-normal process priority for desktop responsiveness")
    ap.add_argument(
        "--optimizer-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offload KahanBF16AdamW state and step math to CPU RAM",
    )
    ap.add_argument(
        "--activation-budget-mb",
        type=float,
        default=None,
        help="VRAM budget in MB for un-checkpointed transformer blocks (0 checkpoints all blocks)",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from <run-dir>/ncm_latest.pt if present")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def train_ncm(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = OmegaConf.load(cfg_path)
    s2_cfg = OmegaConf.to_container(cfg.get("train", {}).get("stage2", {}), resolve=True) or {}

    raw_bucket = getattr(args, "prefix_bucket_size", None)
    prefix_bucket_size = int(raw_bucket if raw_bucket is not None else s2_cfg.get("prefix_bucket_size", 256))
    raw_vram_cap = getattr(args, "max_reserved_vram_mb", None)
    max_reserved_vram_mb = float(raw_vram_cap if raw_vram_cap is not None else s2_cfg.get("max_reserved_vram_mb", 0.0))
    raw_trim_cd = getattr(args, "vram_trim_cooldown_sec", None)
    vram_trim_cooldown_sec = float(
        raw_trim_cd if raw_trim_cd is not None else s2_cfg.get("vram_trim_cooldown_sec", 60.0)
    )
    raw_xpu_frac = getattr(args, "xpu_memory_fraction", None)
    xpu_memory_fraction = float(raw_xpu_frac if raw_xpu_frac is not None else s2_cfg.get("xpu_memory_fraction", 0.0))
    raw_yield_ms = getattr(args, "desktop_yield_ms", None)
    desktop_yield_ms = float(raw_yield_ms if raw_yield_ms is not None else s2_cfg.get("desktop_yield_ms", 0.0))
    raw_recycle_every = getattr(args, "recycle_every", None)
    recycle_every = max(
        0, int(raw_recycle_every if raw_recycle_every is not None else s2_cfg.get("recycle_every", 0))
    )
    raw_max_commit = getattr(args, "max_process_commit_gb", None)
    max_process_commit_gb = float(
        raw_max_commit if raw_max_commit is not None else s2_cfg.get("max_process_commit_gb", 0.0)
    )
    raw_growth = getattr(args, "max_commit_growth_gb", None)
    max_commit_growth_gb = float(
        raw_growth if raw_growth is not None else s2_cfg.get("max_commit_growth_gb", 0.0)
    )
    can_recycle_exit = bool(getattr(args, "allow_recycle_exit", False)) or (
        os.environ.get("NCM_PIPELINE_LOG_ACTIVE") == "1"
    )
    below_normal_priority = False if bool(getattr(args, "no_below_normal_priority", False)) else bool(s2_cfg.get("below_normal_priority", False))
    raw_cpu_off = getattr(args, "optimizer_cpu_offload", None)
    optimizer_cpu_offload = bool(
        raw_cpu_off if raw_cpu_off is not None else s2_cfg.get("optimizer_cpu_offload", False)
    )
    raw_act_bud = getattr(args, "activation_budget_mb", None)
    activation_budget_mb = max(
        0.0,
        float(raw_act_bud if raw_act_bud is not None else s2_cfg.get("activation_budget_mb", 0.0)),
    )

    device = xpu_compat.resolve_device(args.device)
    if below_normal_priority:
        prio_fn = getattr(xpu_compat, "lower_process_priority_for_desktop", None)
        if callable(prio_fn):
            prio_fn()
    if max_reserved_vram_mb > 0.0 or xpu_memory_fraction > 0.0:
        cap_fn = getattr(xpu_compat, "configure_vram_cap", None)
        if callable(cap_fn):
            cap_fn(device, max_reserved_mb=max_reserved_vram_mb, memory_fraction=xpu_memory_fraction)
    xpu_compat.configure_matmul_precision(device)
    autocast_dtype, autocast_enabled = _resolve_dtype(args.dtype, device)
    xpu_compat.seed_everything(int(args.seed))

    run_dir = Path(args.run_dir)
    export_dir = run_dir / "export"
    run_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "STOP"
    latest_ckpt = run_dir / "ncm_latest.pt"
    best_ckpt = run_dir / "ncm_best.pt"
    export_gen_path = export_dir / "generator.pt"
    export_eval_cfg_path = export_dir / "eval_config.yaml"
    status_path = run_dir / "status.json"

    tok_ckpt_path = Path(args.tokenizer_ckpt) if args.tokenizer_ckpt else (export_dir / "tokenizer.pt")
    if not tok_ckpt_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found at {tok_ckpt_path}")

    memmap_root = Path(args.memmap_root)
    prefix_len = int(cfg.model.prefix_len)
    target_len = int(cfg.tokenizer.vqvae.context_length)
    levels = list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
    concept_len = int(sum(levels))

    wide_dir = memmap_root / "stage2_wide"
    gold_dir = memmap_root / "stage2_gold"
    eval_dir = memmap_root / "eval"

    wide_ds = _MemmapStage2Dataset(wide_dir, prefix_len, target_len, concept_len) if wide_dir.exists() else None
    gold_ds = _MemmapStage2Dataset(gold_dir, prefix_len, target_len, concept_len) if gold_dir.exists() else None
    if wide_ds is None and gold_ds is None:
        raise FileNotFoundError(f"Neither stage2_wide nor stage2_gold found under {memmap_root}")
    if wide_ds is None:
        wide_ds = gold_ds
    if gold_ds is None:
        gold_ds = wide_ds

    eval_ds = _MemmapStage2Dataset(eval_dir, prefix_len, target_len, concept_len) if eval_dir.exists() else None

    batch_size = max(1, int(args.batch_size if args.batch_size is not None else s2_cfg.get("batch_size", 2)))
    grad_accum = max(1, int(args.grad_accum if args.grad_accum is not None else s2_cfg.get("grad_accum", 16)))
    lr = float(args.lr if args.lr is not None else s2_cfg.get("lr", 2e-4))
    min_lr = float(args.min_lr if args.min_lr is not None else s2_cfg.get("min_lr", 2e-5))
    warmup_steps = int(args.warmup_steps if args.warmup_steps is not None else s2_cfg.get("warmup_steps", 2000))
    max_steps = int(args.max_steps if args.max_steps is not None else s2_cfg.get("max_steps", 200000))
    stable_frac = float(s2_cfg.get("stable_frac", 0.8))
    grad_ckpt = bool(s2_cfg.get("gradient_checkpointing", True))
    foreach_adamw = bool(s2_cfg.get("foreach_adamw", False))
    opt_name = str(s2_cfg.get("optimizer", "adamw") or "adamw").strip().lower()
    if opt_name not in ("adamw", "kahan_bf16_adamw"):
        raise ValueError(f"Unsupported stage2 optimizer: {opt_name!r}")
    log_every = max(1, int(args.log_every if args.log_every is not None else s2_cfg.get("log_every", 20)))
    eval_every = max(1, int(args.eval_every if args.eval_every is not None else s2_cfg.get("eval_every", 500)))
    save_every = max(1, int(args.save_every if args.save_every is not None else s2_cfg.get("save_every", 500)))
    export_every = max(1, int(args.export_every if args.export_every is not None else s2_cfg.get("export_every", 1000)))
    raw_mode_names = s2_cfg.get("mode_names") or {}
    mode_names: dict[int, str] = (
        {int(k): str(v) for k, v in raw_mode_names.items()}
        if isinstance(raw_mode_names, dict)
        else {}
    )
    pad_token_id = int(cfg.model.get("prefix_pad_token_id", 50257) or 50257)

    vqvae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True))
    vqvae = VQVAE(**vqvae_cfg)
    tok_sd = torch.load(tok_ckpt_path, map_location="cpu", weights_only=False)
    vqvae.load_state_dict(tok_sd["model"], strict=True)
    del tok_sd
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model)
    if opt_name == "kahan_bf16_adamw":
        vqvae = vqvae.to(dtype=torch.bfloat16)
        net = net.to(dtype=torch.bfloat16)
    elif device.type in ("xpu", "cuda") and autocast_dtype == torch.bfloat16:
        vqvae = vqvae.to(dtype=torch.bfloat16)
    vqvae = vqvae.to(device)
    net = net.to(device)
    net._prefix_bucket_size = prefix_bucket_size  # type: ignore[attr-defined]
    net._cpu_prefix_trimmed = True  # type: ignore[attr-defined]

    if grad_ckpt:
        _enable_gradient_checkpointing(net, activation_budget_mb=activation_budget_mb)
        print(
            f"[ncm] gradient checkpointing enabled on transformer.blocks (act_budget_mb={activation_budget_mb:.0f})",
            flush=True,
        )

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    if opt_name == "kahan_bf16_adamw":
        optimizer: torch.optim.Optimizer = KahanBF16AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            cpu_offload=optimizer_cpu_offload,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            foreach=foreach_adamw,
        )

    unigram_datasets = [ds for ds in (wide_ds, gold_ds) if ds is not None]
    token_unigram, level0_unigram = _load_or_build_unigrams(
        memmap_root,
        in_vocab_size=int(cfg.tokenizer.vqvae.in_vocab_size),
        l0_vocab_size=int(cfg.tokenizer.vqvae.vqvae_vocab_size[0]),
        datasets=list({id(d): d for d in unigram_datasets}.values()),
    )

    start_step = 0
    best_val_loss = float("inf")
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model"], strict=True)
        if hasattr(optimizer, "sync_cpu_shadows"):
            optimizer.sync_cpu_shadows()
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        _restore_rng_state(ckpt.get("rng"), device)
        print(f"[ncm] resumed from {latest_ckpt} at step={start_step}/{max_steps} best_val={best_val_loss:.4f}", flush=True)
        del ckpt
        gc.collect()

    _export_eval_config(cfg_path, cfg, export_eval_cfg_path)

    wide_bs = min(batch_size, len(wide_ds))
    gold_bs = min(batch_size, len(gold_ds))
    wide_loader = DataLoader(wide_ds, batch_size=wide_bs, shuffle=True, drop_last=False, collate_fn=_collate_stage2)
    gold_loader = DataLoader(gold_ds, batch_size=gold_bs, shuffle=True, drop_last=False, collate_fn=_collate_stage2)
    eval_loader = (
        DataLoader(eval_ds, batch_size=min(batch_size, len(eval_ds)), shuffle=False, drop_last=False, collate_fn=_collate_stage2)
        if eval_ds is not None
        else None
    )

    wide_iter = iter(wide_loader)
    gold_iter = iter(gold_loader)

    def _next_batch(split_name: str):
        nonlocal wide_iter, gold_iter
        if split_name == "gold":
            try:
                return next(gold_iter)
            except StopIteration:
                gold_iter = iter(gold_loader)
                return next(gold_iter)
        try:
            return next(wide_iter)
        except StopIteration:
            wide_iter = iter(wide_loader)
            return next(wide_iter)

    trim_fn = getattr(xpu_compat, "trim_vram_cache_if_needed", None)

    def _maybe_trim_vram() -> None:
        if max_reserved_vram_mb > 0.0 and callable(trim_fn):
            try:
                trim_fn(device, max_reserved_vram_mb, min_interval_sec=vram_trim_cooldown_sec)
            except TypeError:
                trim_fn(device, max_reserved_vram_mb)

    def _export_generator(
        target_path: Path,
        step_num: int,
        model_cpu_sd: dict[str, Any] | None = None,
    ) -> None:
        model_sd = model_cpu_sd if model_cpu_sd is not None else _snapshot_to_cpu(net.state_dict())
        _atomic_save(
            {
                "model": model_sd,
                "inference": {
                    "token_unigram": token_unigram,
                    "level0_unigram": level0_unigram,
                },
                "step": step_num,
            },
            target_path,
        )

    def _write_status(
        step_num: int,
        state_name: str,
        phase: str,
        cur_lr: float,
        split_name: str,
        train_loss_val: float,
        val_loss_val: float | None = None,
        l0_acc_val: float | None = None,
        per_mode_val: dict[str, float] | None = None,
    ) -> None:
        _atomic_write_json(
            {
                "stage": "train_ncm",
                "state": state_name,
                "step": step_num,
                "max_steps": max_steps,
                "continuous": bool(args.continuous),
                "phase": phase,
                "split": split_name,
                "lr": cur_lr,
                "train_loss": train_loss_val,
                "val_loss": val_loss_val,
                "l0_acc": l0_acc_val,
                "per_mode_loss": per_mode_val,
                "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
                "vram": xpu_compat.get_vram_stats_mb(device),
                "updated_at": time.time(),
            },
            status_path,
        )

    def _save_state(
        step_num: int,
        state_name: str,
        phase: str,
        cur_lr: float,
        split_name: str,
        train_loss_val: float,
        val_loss_val: float | None = None,
        l0_acc_val: float | None = None,
        per_mode_val: dict[str, float] | None = None,
        model_cpu_sd: dict[str, Any] | None = None,
        opt_cpu_sd: dict[str, Any] | None = None,
    ) -> None:
        model_sd = model_cpu_sd if model_cpu_sd is not None else _snapshot_to_cpu(net.state_dict())
        opt_sd = opt_cpu_sd if opt_cpu_sd is not None else _snapshot_to_cpu(optimizer.state_dict())
        _atomic_save(
            {
                "model": model_sd,
                "optimizer": opt_sd,
                "step": step_num,
                "best_val_loss": best_val_loss,
                "rng": _capture_rng_state(device),
            },
            latest_ckpt,
        )
        _write_status(
            step_num,
            state_name,
            phase,
            cur_lr,
            split_name,
            train_loss_val,
            val_loss_val,
            l0_acc_val,
            per_mode_val,
        )

    def _cleanup_and_shutdown() -> None:
        nonlocal net, vqvae, optimizer, trainable_params
        net = None  # type: ignore[assignment]
        vqvae = None  # type: ignore[assignment]
        optimizer = None  # type: ignore[assignment]
        trainable_params = []
        xpu_compat.shutdown_device_cleanly(device)

    print(
        f"[ncm] device={device} autocast={autocast_enabled} dtype={str(autocast_dtype).split('.')[-1]} "
        f"optimizer={opt_name} cpu_offload={optimizer_cpu_offload} act_budget_mb={activation_budget_mb:.0f} "
        f"wide={len(wide_ds)} gold={len(gold_ds)} eval={len(eval_ds) if eval_ds else 0} "
        f"bs={batch_size}x{grad_accum} steps={start_step}->{max_steps} continuous={args.continuous} "
        f"bucket={prefix_bucket_size} vram_cap_mb={max_reserved_vram_mb:.0f} yield_ms={desktop_yield_ms:.0f}",
        flush=True,
    )

    if not args.continuous and start_step >= max_steps:
        phase, cur_lr, split_name = wsd_schedule(
            start_step, max_steps, warmup_steps, stable_frac, lr, min_lr, continuous=False
        )
        _export_generator(export_gen_path, start_step)
        _write_status(start_step, "completed", phase, cur_lr, split_name, 0.0)
        print(f"[ncm] already at step {start_step} >= {max_steps}; ensured {export_gen_path}", flush=True)
        _cleanup_and_shutdown()
        return 0

    net.train()
    last_train_loss = 0.0
    prev_phase = ""
    t_window = time.perf_counter()
    step = start_step
    baseline_commit_gb: float = 0.0

    while args.continuous or step < max_steps:
        phase, cur_lr, split_name = wsd_schedule(
            step, max_steps, warmup_steps, stable_frac, lr, min_lr, continuous=bool(args.continuous)
        )
        if phase != prev_phase:
            print(f"[ncm] step {step}: entering WSD phase={phase} split=stage2_{split_name} lr={cur_lr:.2e}", flush=True)
            prev_phase = phase

        if stop_file.exists():
            print(f"[ncm] STOP sentinel detected before step {step}; saving and exiting", flush=True)
            model_cpu_sd = _snapshot_to_cpu(net.state_dict())
            opt_cpu_sd = _snapshot_to_cpu(optimizer.state_dict())
            _export_generator(export_gen_path, step, model_cpu_sd=model_cpu_sd)
            _save_state(
                step,
                "stopped",
                phase,
                cur_lr,
                split_name,
                last_train_loss,
                model_cpu_sd=model_cpu_sd,
                opt_cpu_sd=opt_cpu_sd,
            )
            _cleanup_and_shutdown()
            return 0

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum):
            if stop_file.exists():
                optimizer.zero_grad(set_to_none=True)
                print(f"[ncm] STOP sentinel detected mid-step {step}; saving and exiting", flush=True)
                model_cpu_sd = _snapshot_to_cpu(net.state_dict())
                opt_cpu_sd = _snapshot_to_cpu(optimizer.state_dict())
                _export_generator(export_gen_path, step, model_cpu_sd=model_cpu_sd)
                _save_state(
                    step,
                    "stopped",
                    phase,
                    cur_lr,
                    split_name,
                    last_train_loss,
                    model_cpu_sd=model_cpu_sd,
                    opt_cpu_sd=opt_cpu_sd,
                )
                _cleanup_and_shutdown()
                return 0

            prefix_ids, target_ids, gt_idx, _ = _next_batch(split_name)
            if not getattr(net, "use_ctx_encoder", False) and prefix_bucket_size > 0:
                prefix_ids = _trim_prefix_to_bucket(
                    prefix_ids, pad_token_id, max(prefix_bucket_size, 64), int(net.levels[0])
                )
            prefix_ids = prefix_ids.to(device, non_blocking=False)
            target_ids = target_ids.to(device, non_blocking=False)
            if gt_idx is not None:
                gt_idx = gt_idx.to(device, non_blocking=False)

            loss, _, _ = _forward_ncm_step(
                net,
                vqvae,
                prefix_ids,
                target_ids,
                gt_idx,
                pad_token_id,
                device,
                autocast_dtype,
                autocast_enabled,
            )
            (loss / float(grad_accum)).backward()
            step_loss += float(loss.item()) / float(grad_accum)
            del prefix_ids, target_ids, gt_idx, loss
            if desktop_yield_ms > 0.0 and device.type in ("xpu", "cuda"):
                xpu_compat.synchronize(device)
                time.sleep(desktop_yield_ms / 1000.0)

        grad_norm, grad_ok = _clip_grad_norm_fp32(trainable_params, 1.0)
        next_step = step + 1
        if (not grad_ok) or (not math.isfinite(step_loss)):
            print(
                f"[ncm] WARNING: non-finite loss={step_loss} or grad_norm={grad_norm} at step {next_step}; skipping optimizer.step()",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            last_train_loss = step_loss
            _maybe_trim_vram()

        if next_step % log_every == 0 or (not args.continuous and next_step == max_steps) or next_step == start_step + 1:
            dt = max(1e-6, time.perf_counter() - t_window)
            t_window = time.perf_counter()
            vram = xpu_compat.get_vram_stats_mb(device)
            commit_gb = float(getattr(xpu_compat, "get_process_commit_gb", lambda: 0.0)())
            commit_str = f" | commit={commit_gb:.1f}GB" if commit_gb > 0.0 else ""
            print(
                f"[ncm] step {next_step}/{max_steps} | phase={phase} ({split_name}) | "
                f"lr={cur_lr:.2e} | loss={step_loss:.4f} | gnorm={grad_norm:.3f} | vram={vram['allocated_mb']:.0f}MB{commit_str} | {dt:.2f}s",
                flush=True,
            )

        val_loss_val: float | None = None
        l0_acc_val: float | None = None
        per_mode_val: dict[str, float] | None = None
        is_final = (not args.continuous) and (next_step >= max_steps)

        model_cpu_sd: dict[str, Any] | None = None
        opt_cpu_sd: dict[str, Any] | None = None
        exported_gen_this_step = False
        saved_state_this_step = False

        def _get_model_cpu_sd() -> dict[str, Any]:
            nonlocal model_cpu_sd
            if model_cpu_sd is None:
                model_cpu_sd = _snapshot_to_cpu(net.state_dict())
            return model_cpu_sd

        def _get_opt_cpu_sd() -> dict[str, Any]:
            nonlocal opt_cpu_sd
            if opt_cpu_sd is None:
                opt_cpu_sd = _snapshot_to_cpu(optimizer.state_dict())
            return opt_cpu_sd

        if next_step % eval_every == 0 or is_final:
            if eval_loader is not None:
                val_loss_val, l0_acc_val, per_mode_val = _evaluate_ncm(
                    net,
                    vqvae,
                    eval_loader,
                    pad_token_id,
                    device,
                    autocast_dtype,
                    autocast_enabled,
                    desktop_yield_ms=desktop_yield_ms,
                    mode_names=mode_names,
                )
                _maybe_trim_vram()
                mode_str = " ".join(f"{k}={v:.4f}" for k, v in per_mode_val.items())
                print(
                    f"[ncm] eval step {next_step}: val_loss={val_loss_val:.4f} L0_acc={l0_acc_val:.3f} | {mode_str}",
                    flush=True,
                )
                score = val_loss_val
            else:
                score = step_loss

            if score < best_val_loss:
                best_val_loss = score
                _atomic_save(
                    {
                        "model": _get_model_cpu_sd(),
                        "optimizer": _get_opt_cpu_sd(),
                        "step": next_step,
                        "best_val_loss": best_val_loss,
                        "rng": _capture_rng_state(device),
                    },
                    best_ckpt,
                )
                _export_generator(export_gen_path, next_step, model_cpu_sd=_get_model_cpu_sd())
                exported_gen_this_step = True

        if next_step % export_every == 0 or is_final:
            if not exported_gen_this_step:
                _export_generator(export_gen_path, next_step, model_cpu_sd=_get_model_cpu_sd())
                exported_gen_this_step = True
            if args.continuous and next_step > max_steps:
                next_phase, _, _ = wsd_schedule(
                    next_step, max_steps, warmup_steps, stable_frac, lr, min_lr, continuous=True
                )
                if next_phase != phase and "anneal" in phase:
                    snap_name = f"{phase}_step_{next_step}.pt"
                    _export_generator(export_dir / snap_name, next_step, model_cpu_sd=_get_model_cpu_sd())

        if next_step % save_every == 0 or is_final:
            _save_state(
                next_step,
                "completed" if is_final else "running",
                phase,
                cur_lr,
                split_name,
                step_loss,
                val_loss_val,
                l0_acc_val,
                per_mode_val,
                model_cpu_sd=_get_model_cpu_sd(),
                opt_cpu_sd=_get_opt_cpu_sd(),
            )
            saved_state_this_step = True

        if (model_cpu_sd is not None or opt_cpu_sd is not None) and not stop_file.exists():
            model_cpu_sd = None
            opt_cpu_sd = None
            gc.collect()

        if desktop_yield_ms > 0.0 and device.type in ("xpu", "cuda"):
            xpu_compat.synchronize(device)
            time.sleep(desktop_yield_ms / 1000.0)

        if stop_file.exists():
            print(f"[ncm] STOP sentinel detected after step {next_step}; saving and exiting", flush=True)
            if not exported_gen_this_step:
                _export_generator(export_gen_path, next_step, model_cpu_sd=_get_model_cpu_sd())
            if not saved_state_this_step:
                _save_state(
                    next_step,
                    "stopped",
                    phase,
                    cur_lr,
                    split_name,
                    step_loss,
                    val_loss_val,
                    l0_acc_val,
                    per_mode_val,
                    model_cpu_sd=_get_model_cpu_sd(),
                    opt_cpu_sd=_get_opt_cpu_sd(),
                )
            else:
                _write_status(
                    next_step,
                    "stopped",
                    phase,
                    cur_lr,
                    split_name,
                    step_loss,
                    val_loss_val,
                    l0_acc_val,
                    per_mode_val,
                )
            _cleanup_and_shutdown()
            return 0

        if can_recycle_exit and not is_final:
            steps_since_start = next_step - start_step
            commit_gb = float(getattr(xpu_compat, "get_process_commit_gb", lambda: 0.0)())
            if steps_since_start == 3 and commit_gb > 0.0 and baseline_commit_gb <= 0.0:
                baseline_commit_gb = commit_gb
            due_steps = (recycle_every > 0) and (steps_since_start >= recycle_every) and (next_step % recycle_every == 0)
            due_abs_commit = (max_process_commit_gb > 0.0) and (commit_gb >= max_process_commit_gb) and (steps_since_start >= 25)
            due_growth_commit = (
                (max_commit_growth_gb > 0.0)
                and (baseline_commit_gb > 0.0)
                and ((commit_gb - baseline_commit_gb) >= max_commit_growth_gb)
                and (steps_since_start >= 25)
            )
            due_commit = due_abs_commit or due_growth_commit
            if due_steps or due_commit:
                print(
                    f"[ncm] recycling worker at step {next_step} "
                    f"(steps_run={steps_since_start}, commit_gb={commit_gb:.2f}, baseline_gb={baseline_commit_gb:.2f}) to release native driver memory",
                    flush=True,
                )
                if not exported_gen_this_step:
                    _export_generator(export_gen_path, next_step, model_cpu_sd=_get_model_cpu_sd())
                if saved_state_this_step:
                    _write_status(
                        next_step,
                        "recycling",
                        phase,
                        cur_lr,
                        split_name,
                        step_loss,
                        val_loss_val,
                        l0_acc_val,
                        per_mode_val,
                    )
                else:
                    _save_state(
                        next_step,
                        "recycling",
                        phase,
                        cur_lr,
                        split_name,
                        step_loss,
                        val_loss_val,
                        l0_acc_val,
                        per_mode_val,
                        model_cpu_sd=_get_model_cpu_sd(),
                        opt_cpu_sd=_get_opt_cpu_sd(),
                    )
                model_cpu_sd = None
                opt_cpu_sd = None
                gc.collect()
                _cleanup_and_shutdown()
                return 0

        model_cpu_sd = None
        opt_cpu_sd = None
        step = next_step

    _cleanup_and_shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train_ncm(args)


if __name__ == "__main__":
    raise SystemExit(main())

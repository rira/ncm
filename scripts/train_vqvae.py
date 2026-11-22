from __future__ import annotations

import argparse
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
from models.vqvae import VQVAE


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


class _MemmapStage1Dataset(Dataset):
    def __init__(self, split_dir: Path, block_len: int, bin_name: str | None = None):
        self.split_dir = Path(split_dir)
        meta_path = self.split_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.block_len = int(meta.get("block_len") or meta.get("seq_len") or meta.get("target_len") or block_len)
        dtype_name = meta.get("dtype") or "uint16"
        self.np_dtype = np.dtype(dtype_name)

        candidates = []
        if bin_name:
            candidates.append(self.split_dir / bin_name)
        if meta.get("blocks_file"):
            candidates.append(self.split_dir / str(meta["blocks_file"]))
        if meta.get("target_file"):
            candidates.append(self.split_dir / str(meta["target_file"]))
        candidates.extend([
            self.split_dir / f"blocks_{self.block_len}.bin",
            self.split_dir / "blocks_256.bin",
            self.split_dir / f"target_{self.block_len}.bin",
            self.split_dir / "blocks.bin",
        ])
        bin_path = next((p for p in candidates if p.exists()), None)
        if bin_path is None:
            raise FileNotFoundError(f"No stage1 memmap binary found under {self.split_dir}")

        file_bytes = bin_path.stat().st_size
        row_bytes = self.block_len * self.np_dtype.itemsize
        inferred_n = file_bytes // row_bytes
        if "dtype" not in meta and file_bytes % (self.block_len * 2) != 0 and file_bytes % (self.block_len * 4) == 0:
            self.np_dtype = np.dtype("int32")
            row_bytes = self.block_len * 4
            inferred_n = file_bytes // row_bytes

        n_samples = int(
            meta.get("num_samples")
            or meta.get("n_samples")
            or meta.get("count")
            or meta.get("num_blocks")
            or inferred_n
        )
        n_samples = min(n_samples, inferred_n) if inferred_n > 0 else 0
        if n_samples <= 0:
            raise ValueError(f"Empty stage1 dataset at {bin_path}")
        self.num_samples = n_samples
        self._mm = np.memmap(bin_path, dtype=self.np_dtype, mode="r", shape=(self.num_samples, self.block_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self._mm[idx], dtype=np.int64))


def _load_stage1_dataset(split_dir: Path, block_len: int) -> Dataset:
    try:
        from data_pipeline.memmap_dataset import Stage1SpecMemmapDataset  # type: ignore

        return Stage1SpecMemmapDataset(str(split_dir))
    except Exception:
        return _MemmapStage1Dataset(split_dir, block_len=block_len)


def _load_stage1_val_dataset(memmap_root: Path, block_len: int) -> Dataset | None:
    for candidate_dir, bin_name in (
        (memmap_root / "stage1_vqvae", f"val_blocks_{block_len}.bin"),
        (memmap_root / "eval", f"target_{block_len}.bin"),
    ):
        if (candidate_dir / bin_name).exists():
            try:
                return _MemmapStage1Dataset(candidate_dir, block_len=block_len, bin_name=bin_name)
            except Exception:
                continue
    return None


def _maybe_init_gpt2_wte(vqvae: VQVAE, model_id: str, enabled: bool) -> None:
    if not enabled:
        print("[vqvae] GPT-2 encoder.wte initialization disabled (--no-init-gpt2-wte)", flush=True)
        return
    try:
        from transformers import GPT2Model

        gpt2 = GPT2Model.from_pretrained(model_id)
        src = gpt2.wte.weight.data.detach().float()
    except Exception as exc:
        print(f"[vqvae] GPT-2 encoder.wte init skipped (offline or unavailable: {exc})", flush=True)
        return

    with torch.no_grad():
        dst = vqvae.encoder.wte.weight.data
        n_tok = min(src.shape[0], dst.shape[0])
        d_src = src.shape[1]
        d_dst = dst.shape[1]
        if d_src == d_dst:
            proj = src[:n_tok]
        elif d_src > d_dst:
            centered = src[:n_tok] - src[:n_tok].mean(dim=0, keepdim=True)
            u, s, _ = torch.linalg.svd(centered, full_matrices=False)
            proj = u[:, :d_dst] * s[:d_dst]
            target_norm = src[:n_tok].norm(dim=1).mean().clamp(min=1e-6) * math.sqrt(d_dst / d_src)
            cur_norm = proj.norm(dim=1).mean().clamp(min=1e-6)
            proj = proj * (target_norm / cur_norm)
        else:
            proj = torch.zeros(n_tok, d_dst, dtype=torch.float32)
            proj[:, :d_src] = src[:n_tok]
        dst[:n_tok].copy_(proj.to(dtype=dst.dtype, device=dst.device))
    print(f"[vqvae] initialized encoder.wte[:{n_tok}] from {model_id} ({d_src}->{d_dst})", flush=True)


def _cosine_lr(step: int, max_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    if max_steps <= warmup_steps:
        return lr
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _reset_codebook_hits(vqvae: VQVAE) -> None:
    num_levels = len(vqvae.vq.levels)
    for i in range(num_levels):
        buf = getattr(vqvae.vq, f"codebook_hit_{i}", None)
        if buf is not None:
            buf.zero_()


@torch.no_grad()
def _evaluate_vqvae(
    vqvae: VQVAE,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool,
    pad_token_id: int,
    max_batches: int = 16,
) -> tuple[float, float]:
    was_training = vqvae.training
    vqvae.eval()
    recon_sum = 0.0
    vq_sum = 0.0
    count = 0
    for idx, batch in enumerate(val_loader):
        if idx >= max_batches:
            break
        batch = batch.to(device, non_blocking=False)
        with xpu_compat.autocast_context(device, dtype=autocast_dtype, enabled=autocast_enabled):
            logits, vq_loss, _, _, _ = vqvae(batch)
            recon_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                batch.reshape(-1),
                ignore_index=pad_token_id,
            )
        recon_sum += float(recon_loss.item())
        vq_sum += float(vq_loss.float().item())
        count += 1
    if was_training:
        vqvae.train()
    if count == 0:
        return float("inf"), float("inf")
    return (recon_sum + vq_sum) / count, recon_sum / count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 1 Resumable VQ-VAE Trainer for Intel XPU / CUDA / CPU")
    ap.add_argument("--config", required=True, help="Path to config YAML (e.g. config/train/ncm_500m.yaml)")
    ap.add_argument("--memmap-root", default="data/memmap", help="Root directory containing stage1_vqvae/")
    ap.add_argument("--run-dir", default="runs/ncm_500m", help="Output run directory for checkpoints and status")
    ap.add_argument("--device", default="auto", help="Compute device (auto, xpu, cuda, cpu)")
    ap.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--max-steps", type=int, default=None, help="Override train.stage1.max_steps")
    ap.add_argument("--batch-size", type=int, default=None, help="Override train.stage1.batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="Override train.stage1.grad_accum")
    ap.add_argument("--lr", type=float, default=None, help="Override train.stage1.lr")
    ap.add_argument("--log-every", type=int, default=None, help="Override train.stage1.log_every")
    ap.add_argument("--eval-every", type=int, default=None, help="Override train.stage1.eval_every")
    ap.add_argument("--save-every", type=int, default=None, help="Override train.stage1.save_every")
    ap.add_argument("--resume", action="store_true", help="Resume from <run-dir>/vqvae_latest.pt if present")
    ap.add_argument(
        "--init-gpt2-wte",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize encoder.wte from GPT-2 embeddings on step 0 (skips cleanly if offline)",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def train_vqvae(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(args.config)
    s1_cfg = OmegaConf.to_container(cfg.get("train", {}).get("stage1", {}), resolve=True) or {}

    device = xpu_compat.resolve_device(args.device)
    xpu_compat.configure_matmul_precision(device)
    autocast_dtype, autocast_enabled = _resolve_dtype(args.dtype, device)
    xpu_compat.seed_everything(int(args.seed))

    run_dir = Path(args.run_dir)
    export_dir = run_dir / "export"
    run_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "STOP"
    latest_ckpt = run_dir / "vqvae_latest.pt"
    best_ckpt = run_dir / "vqvae_best.pt"
    export_tok_path = export_dir / "tokenizer.pt"
    status_path = run_dir / "status.json"

    memmap_root = Path(args.memmap_root)
    stage1_dir = memmap_root / "stage1_vqvae"
    block_len = int(cfg.tokenizer.vqvae.context_length)
    train_ds = _load_stage1_dataset(stage1_dir, block_len=block_len)
    val_ds = _load_stage1_val_dataset(memmap_root, block_len=block_len)

    batch_size = max(1, int(args.batch_size if args.batch_size is not None else s1_cfg.get("batch_size", 16)))
    grad_accum = max(1, int(args.grad_accum if args.grad_accum is not None else s1_cfg.get("grad_accum", 1)))
    lr = float(args.lr if args.lr is not None else s1_cfg.get("lr", 3e-4))
    min_lr = float(s1_cfg.get("min_lr", lr * 0.1))
    warmup_steps = int(s1_cfg.get("warmup_steps", 500))
    max_steps = int(args.max_steps if args.max_steps is not None else s1_cfg.get("max_steps", 25000))
    log_every = max(1, int(args.log_every if args.log_every is not None else s1_cfg.get("log_every", 20)))
    eval_every = max(1, int(args.eval_every if args.eval_every is not None else s1_cfg.get("eval_every", 500)))
    save_every = max(1, int(args.save_every if args.save_every is not None else s1_cfg.get("save_every", 500)))
    kmeans_batches = max(1, int(s1_cfg.get("kmeans_init_batches", 8)))
    foreach_adamw = bool(s1_cfg.get("foreach_adamw", False))
    pad_token_id = int(cfg.model.get("prefix_pad_token_id", 50257) or 50257)

    vqvae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True))
    vqvae = VQVAE(**vqvae_cfg).to(device)

    trainable_params = [p for p in vqvae.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=foreach_adamw,
    )

    start_step = 0
    best_loss = float("inf")
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        vqvae.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", float("inf")))
        _restore_rng_state(ckpt.get("rng"), device)
        print(f"[vqvae] resumed from {latest_ckpt} at step={start_step}/{max_steps} best_loss={best_loss:.4f}", flush=True)
    else:
        model_id = str(cfg.tokenizer.pre_tokenizer.get("model_id", "gpt2"))
        _maybe_init_gpt2_wte(vqvae, model_id=model_id, enabled=bool(args.init_gpt2_wte))

    eff_bs = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=eff_bs, shuffle=False, drop_last=False) if val_ds is not None else None

    if start_step == 0 and len(train_ds) > 0:
        vqvae.eval()
        km_chunks = []
        with torch.no_grad():
            for b_idx, batch in enumerate(train_loader):
                if b_idx >= kmeans_batches:
                    break
                batch = batch.to(device)
                with xpu_compat.autocast_context(device, dtype=autocast_dtype, enabled=autocast_enabled):
                    z = vqvae._encode_pre_quant(batch)
                    z0 = vqvae.vq._down_for_quant(z.float(), 0, vqvae.vq.levels[0]).reshape(-1, vqvae.embed_dim)
                km_chunks.append(z0.float())
        if km_chunks:
            samples = torch.cat(km_chunks, dim=0)
            if samples.shape[0] > 0:
                vqvae.vq.kmeans_init_level(0, samples)
                print(f"[vqvae] step-0 kmeans_init_level(0) with {samples.shape[0]} samples", flush=True)

    print(
        f"[vqvae] device={device} autocast={autocast_enabled} dtype={str(autocast_dtype).split('.')[-1]} "
        f"samples={len(train_ds)} bs={eff_bs}x{grad_accum} steps={start_step}->{max_steps}",
        flush=True,
    )

    def _write_status(step_num: int, state_name: str, train_loss_val: float, val_loss_val: float | None = None) -> None:
        _atomic_write_json(
            {
                "stage": "train_vqvae",
                "state": state_name,
                "step": step_num,
                "max_steps": max_steps,
                "train_loss": train_loss_val,
                "val_loss": val_loss_val,
                "best_loss": best_loss if math.isfinite(best_loss) else None,
                "vram": xpu_compat.get_vram_stats_mb(device),
                "updated_at": time.time(),
            },
            status_path,
        )

    def _save_state(step_num: int, state_name: str, train_loss_val: float, val_loss_val: float | None = None) -> None:
        ckpt_obj = {
            "model": vqvae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step_num,
            "best_loss": best_loss,
            "rng": _capture_rng_state(device),
        }
        _atomic_save(ckpt_obj, latest_ckpt)
        _atomic_save({"model": vqvae.state_dict()}, export_tok_path)
        _write_status(step_num, state_name, train_loss_val, val_loss_val)

    def _cleanup_and_shutdown() -> None:
        nonlocal vqvae, optimizer, trainable_params
        vqvae = None  # type: ignore[assignment]
        optimizer = None  # type: ignore[assignment]
        trainable_params = []
        xpu_compat.shutdown_device_cleanly(device)

    if start_step >= max_steps:
        _atomic_save({"model": vqvae.state_dict()}, export_tok_path)
        _write_status(start_step, "completed", 0.0)
        print(f"[vqvae] already at step {start_step} >= {max_steps}; ensured {export_tok_path}", flush=True)
        _cleanup_and_shutdown()
        return 0

    train_iter = iter(train_loader)

    def _next_batch() -> torch.Tensor:
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    vqvae.train()
    last_train_loss = 0.0
    t_window = time.perf_counter()

    for step in range(start_step, max_steps):
        if stop_file.exists():
            print(f"[vqvae] STOP sentinel detected before step {step}; saving and exiting", flush=True)
            _save_state(step, "stopped", last_train_loss)
            _cleanup_and_shutdown()
            return 0

        cur_lr = _cosine_lr(step, max_steps, warmup_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_recon = 0.0
        step_vq = 0.0
        per_level_util: list[float] = []
        global_util = 0.0

        for _ in range(grad_accum):
            if stop_file.exists():
                optimizer.zero_grad(set_to_none=True)
                print(f"[vqvae] STOP sentinel detected mid-step {step}; saving and exiting", flush=True)
                _save_state(step, "stopped", last_train_loss)
                _cleanup_and_shutdown()
                return 0

            batch = _next_batch().to(device, non_blocking=False)
            with xpu_compat.autocast_context(device, dtype=autocast_dtype, enabled=autocast_enabled):
                logits, vq_loss, per_level_util, global_util, _ = vqvae(batch)
                recon_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    batch.reshape(-1),
                    ignore_index=pad_token_id,
                )
                loss = recon_loss + vq_loss.float()

            (loss / float(grad_accum)).backward()
            step_loss += float(loss.item()) / float(grad_accum)
            step_recon += float(recon_loss.item()) / float(grad_accum)
            step_vq += float(vq_loss.float().item()) / float(grad_accum)
            del logits, recon_loss, loss, vq_loss, batch

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        next_step = step + 1
        last_train_loss = step_loss

        if next_step % log_every == 0 or next_step == max_steps or next_step == start_step + 1:
            dt = max(1e-6, time.perf_counter() - t_window)
            t_window = time.perf_counter()
            util_str = ",".join(f"{u:.2f}" for u in per_level_util)
            vram = xpu_compat.get_vram_stats_mb(device)
            print(
                f"[vqvae] step {next_step}/{max_steps} | lr={cur_lr:.2e} | "
                f"loss={step_loss:.4f} (recon={step_recon:.4f}, vq={step_vq:.4f}) | "
                f"util={global_util:.3f} [{util_str}] | vram={vram['allocated_mb']:.0f}MB | {dt:.2f}s",
                flush=True,
            )
            _reset_codebook_hits(vqvae)

        val_loss_val: float | None = None
        if next_step % eval_every == 0 or next_step == max_steps:
            if val_loader is not None:
                val_loss_val, val_recon = _evaluate_vqvae(
                    vqvae, val_loader, device, autocast_dtype, autocast_enabled, pad_token_id
                )
                print(
                    f"[vqvae] eval step {next_step}: val_loss={val_loss_val:.4f} val_recon={val_recon:.4f}",
                    flush=True,
                )
                score = val_loss_val
            else:
                score = step_loss
            if score < best_loss:
                best_loss = score
                _atomic_save(
                    {
                        "model": vqvae.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": next_step,
                        "best_loss": best_loss,
                        "rng": _capture_rng_state(device),
                    },
                    best_ckpt,
                )
                _atomic_save({"model": vqvae.state_dict()}, export_tok_path)

        if next_step % save_every == 0 or next_step == max_steps:
            _save_state(next_step, "completed" if next_step == max_steps else "running", step_loss, val_loss_val)

        if stop_file.exists():
            print(f"[vqvae] STOP sentinel detected after step {next_step}; saving and exiting", flush=True)
            _save_state(next_step, "stopped", step_loss, val_loss_val)
            _cleanup_and_shutdown()
            return 0

    _cleanup_and_shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train_vqvae(args)


if __name__ == "__main__":
    raise SystemExit(main())

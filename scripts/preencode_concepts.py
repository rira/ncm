from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = os.environ.get("NCM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat
from models.vqvae import VQVAE


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _resolve_amp_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    low = str(dtype_arg).strip().lower()
    if low == "auto":
        return torch.bfloat16 if device.type in ("xpu", "cuda") else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if low not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg!r}")
    return mapping[low]


def _load_vqvae(config_path: str, checkpoint_path: str, device: torch.device) -> VQVAE:
    cfg = OmegaConf.load(config_path)
    vqvae = VQVAE(**cfg.tokenizer.vqvae)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    vqvae.load_state_dict(sd, strict=True)
    vqvae = vqvae.to(device).eval()
    return vqvae


def _normalize_unigram(counts: torch.Tensor) -> torch.Tensor:
    counts_cpu = counts.to(device="cpu", dtype=torch.float32)
    total = float(counts_cpu.sum().item())
    if total > 0.0:
        return counts_cpu / total
    n = counts_cpu.numel()
    return torch.full((n,), 1.0 / float(n), dtype=torch.float32)


def preencode_splits(
    config_path: str,
    tokenizer_ckpt: str,
    memmap_root: str | Path,
    splits: list[str],
    batch_size: int = 256,
    device_arg: str = "auto",
    dtype_arg: str = "bfloat16",
) -> dict[str, Any]:
    device = xpu_compat.resolve_device(device_arg)
    xpu_compat.configure_matmul_precision(device)
    amp_dtype = _resolve_amp_dtype(dtype_arg, device)
    amp_enabled = amp_dtype in (torch.bfloat16, torch.float16)

    vqvae = _load_vqvae(config_path, tokenizer_ckpt, device)
    in_vocab_size = int(vqvae.in_vocab_size)
    levels = [int(x) for x in vqvae.vq.levels]
    concept_len = sum(levels)
    l0_len = levels[0]
    l0_vocab_size = int(vqvae.vq.vqvae_vocab_sizes[0])

    token_counts = torch.zeros(in_vocab_size, dtype=torch.int64)
    level0_counts = torch.zeros(l0_vocab_size, dtype=torch.int64)
    root = Path(memmap_root)
    processed_splits: dict[str, int] = {}

    for split in splits:
        split_dir = root / split
        meta_path = split_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing meta.json in split {split_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        num_rows = int(meta["num_rows"])
        target_len = int(meta["target_len"])
        pad_token_id = int(meta.get("pad_token_id", 50257))
        target_name = str(meta.get("target_file") or f"target_{target_len}.bin")
        target_path = split_dir / target_name
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target memmap file: {target_path}")

        gt_filename = f"gt_idx_{concept_len}.bin"
        gt_path = split_dir / gt_filename
        tmp_gt_path = split_dir / f"{gt_filename}.tmp"

        if num_rows == 0:
            with open(tmp_gt_path, "wb"):
                pass
            os.replace(tmp_gt_path, gt_path)
        else:
            targets_mm = np.memmap(target_path, dtype=np.uint16, mode="r", shape=(num_rows, target_len))
            with open(tmp_gt_path, "wb") as out_fh:
                with torch.no_grad():
                    for start in range(0, num_rows, batch_size):
                        end = min(num_rows, start + batch_size)
                        batch_np = np.array(targets_mm[start:end], dtype=np.int64)
                        batch_cpu = torch.from_numpy(batch_np)

                        valid_toks = batch_cpu[(batch_cpu != pad_token_id) & (batch_cpu >= 0) & (batch_cpu < in_vocab_size)]
                        if valid_toks.numel() > 0:
                            token_counts += torch.bincount(valid_toks, minlength=in_vocab_size)

                        batch_dev = batch_cpu.to(device)
                        with xpu_compat.autocast_context(device, dtype=amp_dtype, enabled=amp_enabled):
                            gt_idx = vqvae.encode(batch_dev)

                        gt_cpu = gt_idx.to(device="cpu", dtype=torch.int64)
                        if gt_cpu.shape != (end - start, concept_len):
                            raise ValueError(
                                f"Expected VQVAE.encode shape {(end - start, concept_len)}, got {tuple(gt_cpu.shape)}"
                            )
                        l0_idx = gt_cpu[:, :l0_len].reshape(-1)
                        valid_l0 = l0_idx[(l0_idx >= 0) & (l0_idx < l0_vocab_size)]
                        if valid_l0.numel() > 0:
                            level0_counts += torch.bincount(valid_l0, minlength=l0_vocab_size)

                        gt_cpu.numpy().astype(np.uint16).tofile(out_fh)
            os.replace(tmp_gt_path, gt_path)

        meta["concept_len"] = concept_len
        meta["gt_idx_file"] = gt_filename
        meta["levels"] = levels
        meta["vqvae_vocab_sizes"] = [int(v) for v in vqvae.vq.vqvae_vocab_sizes]
        _atomic_write_json(meta_path, meta)
        processed_splits[split] = num_rows

    token_unigram = _normalize_unigram(token_counts)
    level0_unigram = _normalize_unigram(level0_counts)
    unigrams_path = root / "unigrams.pt"
    tmp_unigrams = root / "unigrams.pt.tmp"
    torch.save(
        {
            "token_unigram": token_unigram,
            "level0_unigram": level0_unigram,
        },
        tmp_unigrams,
    )
    os.replace(tmp_unigrams, unigrams_path)

    vqvae = None
    xpu_compat.shutdown_device_cleanly(device)

    return {
        "concept_len": concept_len,
        "splits": processed_splits,
        "unigrams_path": str(unigrams_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pre-encode VQ-VAE concept indices and unigrams into memmap shards.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--memmap-root", default="data/memmap")
    ap.add_argument("--splits", nargs="+", default=["stage2_wide", "stage2_gold", "eval"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = preencode_splits(
        config_path=args.config,
        tokenizer_ckpt=args.tokenizer_ckpt,
        memmap_root=args.memmap_root,
        splits=list(args.splits),
        batch_size=int(args.batch_size),
        device_arg=args.device,
        dtype_arg=args.dtype,
    )
    print(
        f"[preencode_concepts] concept_len={summary['concept_len']} "
        f"splits={summary['splits']} unigrams={summary['unigrams_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

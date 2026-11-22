import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

ROOT = os.environ.get("NCM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat
from scripts.eval_shard import (
    build_l0_prior,
    device_display_name,
    generate_batch,
    load_allowance,
    load_models,
    parse_adj,
    resolve_dtype,
)
from tokenization.sudoku_tokenizer import SudokuTokenizer

TASK_HF_SUBPATHS = {
    "xsum": "xsum",
    "wmt14": "wmt14",
    "tinystories": "tinystories",
    "owt": "owt",
    "sudoku-easy": "sudoku/easy",
    "sudoku-medium": "sudoku/medium",
    "sudoku-hard": "sudoku/hard",
    "lm1b-small": "lm1b/small",
    "lm1b-medium": "lm1b/medium",
    "lm1b-large": "lm1b/large",
}


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Interactive / single-run coarse-to-fine generation CLI")
    ap.add_argument("--config", default=None, help="eval config path, e.g. config/eval/xsum.yaml")
    ap.add_argument("--task", default=None, help="task shorthand mapping to config/eval/<task>.yaml")
    ap.add_argument("--checkpoint-dir", default="", help="checkpoint directory (default: artifacts/<task>)")
    ap.add_argument("--download", action="store_true", help="download model checkpoints from hazyresearch/ncm-checkpoints")
    ap.add_argument("--prompt", default=None, help="input prompt for conditional models (text or 81-char/piped Sudoku board)")
    ap.add_argument("--num-samples", type=int, default=1, help="number of samples to generate")
    ap.add_argument("--seed", type=int, default=42, help="PRNG seed")
    ap.add_argument("--temp", type=float, default=None, help="sampling temperature override")
    ap.add_argument("--top_k", type=int, default=None, help="top-k sampling override")
    ap.add_argument("--top_p", type=float, default=None, help="nucleus top-p override")
    ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale override")
    ap.add_argument("--freq", type=float, default=None, help="frequency penalty override")
    ap.add_argument("--mu", type=float, default=None, help="frequency deficit override")
    ap.add_argument("--adj", default=None, help="adjacency penalty override (e.g. 5,4,0.5)")
    ap.add_argument("--l0", default="", help="level-0 prior mode: unigram, active, uniform, or smooth:<eps>")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--device", default=None, help="target compute device (e.g. xpu, cuda:0, cpu)")
    ap.add_argument("--warmup", action="store_true", help="run 1 warmup pass before timed generation")
    return ap.parse_args(argv)


def resolve_config_path(config_arg: str | None, task_arg: str | None, root: str = ROOT) -> str:
    if config_arg:
        if os.path.exists(config_arg):
            return config_arg
        rel = os.path.join(root, config_arg)
        if os.path.exists(rel):
            return rel
        return config_arg
    if task_arg:
        name = task_arg[:-5] if task_arg.endswith(".yaml") else task_arg
        return os.path.join(root, "config", "eval", f"{name}.yaml")
    raise ValueError("one of --config or --task is required")


def get_hf_allow_patterns(task: str) -> tuple[str, list[str]]:
    if task.startswith("lm1b-"):
        size = task.split("-", 1)[1]
        subpath = f"lm1b/{size}"
        return subpath, ["lm1b/tokenizer.pt", f"lm1b/{size}/*"]
    subpath = TASK_HF_SUBPATHS.get(task, task)
    return subpath, [f"{subpath}/*"]


def resolve_checkpoint_dir(cfg, checkpoint_dir_arg: str, download: bool, root: str = ROOT) -> str:
    task = str(cfg.task)
    subpath, allow_patterns = get_hf_allow_patterns(task)
    if download:
        from huggingface_hub import snapshot_download

        if not checkpoint_dir_arg:
            local_dir = os.path.join(root, "artifacts")
            target_dir = os.path.join(local_dir, subpath)
        else:
            norm = os.path.normpath(checkpoint_dir_arg)
            suffix = os.path.normpath(subpath)
            if norm == suffix or norm.endswith(os.sep + suffix):
                local_dir = norm[:-len(suffix)].rstrip(os.sep) or "."
                target_dir = norm
            else:
                local_dir = norm
                target_dir = os.path.join(norm, subpath)
        snapshot_download(
            repo_id="hazyresearch/ncm-checkpoints",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
        return target_dir

    if checkpoint_dir_arg:
        return checkpoint_dir_arg

    gen_name = str(cfg.checkpoints.generator)
    for candidate in (
        os.path.join("artifacts", task),
        os.path.join(root, "artifacts", task),
        os.path.join("artifacts", subpath),
        os.path.join(root, "artifacts", subpath),
    ):
        if os.path.exists(os.path.join(candidate, gen_name)):
            return candidate
    return os.path.join("artifacts", task)


def encode_prompt(prompt: str, tk, cfg) -> list[int]:
    prefix_len = int(cfg.model.prefix_len)
    pre_tok = tk.pre_tokenizer
    is_sudoku = isinstance(pre_tok, SudokuTokenizer) or (
        isinstance(getattr(pre_tok, "blank_token_id", None), int)
        and isinstance(getattr(pre_tok, "row_separator_id", None), int)
    )
    if is_sudoku:
        text = prompt.replace(".", "_")
        compact = "".join(ch for ch in text.replace("<BOS>", "B") if not ch.isspace())
        if "|" not in compact and len(compact) == 81:
            text = "|".join(compact[i * 9:(i + 1) * 9] for i in range(9))
        ids = pre_tok.encode(text)
        if len(ids) == prefix_len - 2:
            ids = [pre_tok.bos_token_id] + ids + [pre_tok.row_separator_id]
        elif len(ids) == prefix_len - 1:
            if ids[0] != pre_tok.bos_token_id:
                ids = [pre_tok.bos_token_id] + ids
            else:
                ids = ids + [pre_tok.row_separator_id]
        pad_id = getattr(cfg.model, "prefix_pad_token_id", None)
        fallback_pad = int(pad_id) if pad_id is not None else 0
        return (ids[:prefix_len] + [fallback_pad] * prefix_len)[:prefix_len]

    pad_id = pre_tok.pad_token_id
    if pad_id is None:
        model_pad = getattr(cfg.model, "prefix_pad_token_id", None)
        pad_id = int(model_pad) if model_pad is not None else 0
    ids = pre_tok.encode(prompt)
    return (ids[:prefix_len] + [pad_id] * prefix_len)[:prefix_len]


def format_vram_stats(stats: dict[str, float]) -> str:
    if not stats:
        return "n/a"
    for key in ("peak_allocated_mb", "max_allocated_mb", "peak_mb", "max_memory_allocated_mb", "allocated_mb"):
        if key in stats:
            alloc = stats.get("allocated_mb")
            if alloc is not None and key != "allocated_mb":
                return f"peak={stats[key]:.1f} MB (allocated={alloc:.1f} MB)"
            return f"{key}={stats[key]:.1f} MB"
    return ", ".join(f"{k}={v:.1f} MB" for k, v in stats.items())


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    try:
        config_path = resolve_config_path(args.config, args.task, root=ROOT)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    cfg = OmegaConf.load(config_path)
    gen_cfg = cfg.generation
    pick = lambda cli, key, cast, default: cast(cli if cli is not None else gen_cfg.get(key, default))
    temp = pick(args.temp, "temperature", float, 1.0)
    top_k = pick(args.top_k, "top_k", int, 1)
    top_p = pick(args.top_p, "top_p", float, 0.0)
    cfg_scale = pick(args.cfg, "cfg_scale", float, 0.0)
    freq = pick(args.freq, "frequency_penalty", float, 0.0)
    mu = pick(args.mu, "frequency_deficit", float, 0.0)
    adj_spec = str(args.adj if args.adj is not None else gen_cfg.get("adjacency_penalty", "") or "")
    adj = parse_adj(adj_spec)
    cut_at_eos = bool(gen_cfg.get("cut_at_eos", True))

    checkpoint_dir = resolve_checkpoint_dir(cfg, args.checkpoint_dir, download=args.download, root=ROOT)

    device = xpu_compat.resolve_device(args.device)
    xpu_compat.configure_matmul_precision(device)
    dtype = resolve_dtype(args.dtype, device)
    dev_name = device_display_name(device)
    dtype_name = str(dtype).split(".")[-1]

    if device.type == "xpu":
        notice = xpu_compat.check_ollama_vram_notice()
        if notice:
            print(notice, flush=True)

    tk, net = load_models(cfg, checkpoint_dir, device)
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    allow = load_allowance(checkpoint_dir, str(gen_cfg.frequency_allowance), device) if gen_cfg.get("frequency_allowance") else None
    l0_prior = build_l0_prior(checkpoint_dir, gen_cfg.get("level0_unigram"), args.l0 or "unigram", device)

    split = cfg.eval.get("split", "test") if cfg.get("eval") else "test"
    uncond = not cfg.data or not cfg.data.get(split)
    batch_size = max(1, int(args.num_samples))

    if uncond:
        if args.prompt is not None:
            raise SystemExit(f"error: task {cfg.task!r} is unconditional and does not accept --prompt")
        l0_len = tk.vqvae.vq.levels[0]
        x = torch.full((batch_size, 1), eos, dtype=torch.long, device=device)
    else:
        if args.prompt is None:
            raise SystemExit(f"error: task {cfg.task!r} requires --prompt")
        prefix_ids = encode_prompt(args.prompt, tk, cfg)
        x = torch.tensor([prefix_ids] * batch_size, dtype=torch.long, device=device)
        l0_len = 0

    def _sample_l0():
        if not uncond or l0_prior is None:
            return None
        return torch.multinomial(l0_prior.cpu(), batch_size * l0_len, replacement=True).view(batch_size, l0_len).to(device)

    if args.warmup:
        xpu_compat.seed_everything(args.seed)
        generate_batch(
            net,
            tk,
            x,
            device=device,
            dtype=dtype,
            l0=_sample_l0(),
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
            eos=eos,
            pad=pad,
            freq=freq,
            mu=mu,
            allow=allow,
            adj=adj,
            cut_at_eos=cut_at_eos,
        )
        xpu_compat.synchronize(device)
        xpu_compat.empty_cache(device)

    xpu_compat.seed_everything(args.seed)
    l0 = _sample_l0()
    xpu_compat.synchronize(device)
    t0 = time.perf_counter()
    outputs, _ = generate_batch(
        net,
        tk,
        x,
        device=device,
        dtype=dtype,
        l0=l0,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        eos=eos,
        pad=pad,
        freq=freq,
        mu=mu,
        allow=allow,
        adj=adj,
        cut_at_eos=cut_at_eos,
    )
    xpu_compat.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    vram_stats = xpu_compat.get_vram_stats_mb(device)

    print(f"[generate] task={cfg.task} device={device} ({dev_name}) dtype={dtype_name}", flush=True)
    print(f"[generate] latency={elapsed_ms:.1f} ms | vram={format_vram_stats(vram_stats)}", flush=True)
    for idx, text in enumerate(outputs):
        if len(outputs) == 1:
            print(f"--- Output ---\n{text}", flush=True)
        else:
            print(f"--- Sample {idx + 1}/{len(outputs)} ---\n{text}", flush=True)


if __name__ == "__main__":
    main()

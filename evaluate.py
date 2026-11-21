import argparse
import os
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="one generation shard per GPU/device, then merge and score")
    ap.add_argument("--config", required=True, help="eval config, e.g. config/eval/xsum.yaml")
    ap.add_argument("--checkpoint-dir", required=True, help="directory with the files named under checkpoints:")
    ap.add_argument("--data-dir", default="", help="directory with the files named under data: (not needed for unconditional tasks)")
    ap.add_argument("--out", default="", help="output directory (default eval/<task>)")
    ap.add_argument("--gpus", default="", help="comma-separated device ids (default: all visible)")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n inputs")
    ap.add_argument("--device", default="", help="target compute device (e.g. xpu, cuda:0, cpu)")
    ap.add_argument("--dtype", default="", help="compute dtype override (auto, bfloat16, float16, float32)")
    return ap.parse_known_args(argv)


def resolve_worker_plan(gpus_arg: str, device_arg: str) -> tuple[str, list[str]]:
    dev = device_arg.strip()
    if dev and not gpus_arg:
        if dev.lower() == "cpu":
            return "cpu", ["0"]
        if ":" in dev:
            dev_backend, dev_idx = dev.split(":", 1)
            return dev_backend.lower(), [dev_idx]
    backend_type, gpus = xpu_compat.get_worker_device_ids(gpus_arg)
    if dev:
        dev_backend = dev.split(":", 1)[0].lower()
        if dev_backend in ("cuda", "xpu", "cpu"):
            backend_type = dev_backend
    return backend_type, gpus


def build_worker_env(
    base_env,
    backend_type: str,
    gpu: str,
    device_arg: str = "",
) -> dict[str, str]:
    env = xpu_compat.configure_worker_env(base_env, backend_type, gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCM_ROOT"] = ROOT
    dev = device_arg.strip()
    if dev:
        isolated = (
            env.get("CUDA_VISIBLE_DEVICES") == gpu
            or env.get("ZE_AFFINITY_MASK") == gpu
            or env.get("ONEAPI_DEVICE_SELECTOR", "").endswith(f":{gpu}")
        )
        env["NCM_DEVICE"] = dev.split(":", 1)[0].lower() if (":" in dev and isolated) else dev
    return env


def build_merge_env(
    base_env,
    backend_type: str,
    first_gpu: str,
    device_arg: str = "",
) -> dict[str, str]:
    env = build_worker_env(base_env, backend_type, first_gpu, device_arg=device_arg)
    if not env.get("NCM_DEVICE"):
        env["NCM_DEVICE"] = backend_type if backend_type in ("xpu", "cuda") else "cpu"
    return env


def main():
    args, passthrough = parse_args()
    cfg = OmegaConf.load(args.config)
    backend_type, gpus = resolve_worker_plan(args.gpus, args.device)
    if not gpus:
        sys.exit("no compute device visible; pass --gpus or --device cpu")
    out = args.out or os.path.join("eval", cfg.task)
    os.makedirs(out, exist_ok=True)

    if backend_type == "xpu":
        notice = xpu_compat.check_ollama_vram_notice()
        if notice:
            print(notice, flush=True)

    common = (
        ["--config", args.config, "--checkpoint-dir", args.checkpoint_dir]
        + (["--data-dir", args.data_dir] if args.data_dir else [])
        + ["--nshards", str(len(gpus))]
        + (["--n", str(args.n)] if args.n else [])
        + (["--dtype", args.dtype] if args.dtype else [])
        + passthrough
    )
    procs = []
    logs = []
    try:
        for k, gpu in enumerate(gpus):
            env = build_worker_env(os.environ, backend_type, gpu, device_arg=args.device)
            log = open(os.path.join(out, f"shard_{k}.log"), "w")
            logs.append(log)
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "scripts", "eval_shard.py"), *common,
                 "--shard", str(k), "--out", os.path.join(out, f"shard_{k}.pkl")],
                env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
        print(f"[evaluate] {len(gpus)} shard(s) on {backend_type.upper()} device(s) {','.join(gpus)} -> {out}/", flush=True)
        failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    finally:
        for log in logs:
            log.close()
    if failed:
        sys.exit(f"shard(s) {failed} failed; see {out}/shard_*.log")

    merge_env = build_merge_env(os.environ, backend_type, gpus[0], device_arg=args.device)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "eval_merge.py"), "--config", args.config,
         "--glob", os.path.join(out, "shard_*.pkl")],
        check=True, cwd=ROOT, env=merge_env)


if __name__ == "__main__":
    main()

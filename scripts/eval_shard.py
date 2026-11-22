import argparse
import os
import pickle
import sys

import torch
from omegaconf import OmegaConf

ROOT = os.environ.get("NCM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--adj", default=None)
    ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps> (mix eps of uniform into the unigram)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type in ("xpu", "cuda") else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_arg!r}")
    return mapping[dtype_arg]


def get_stat_dtype(device: torch.device) -> torch.dtype:
    return torch.float64 if xpu_compat.supports_fp64(device) else torch.float32


def device_display_name(device: torch.device) -> str:
    try:
        if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "get_device_name"):
            return str(torch.xpu.get_device_name(device))
        if device.type == "cuda" and hasattr(torch.cuda, "get_device_name"):
            return str(torch.cuda.get_device_name(device))
    except Exception:
        pass
    return str(device)


def load_allowance(
    checkpoint_dir: str,
    spec: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype if dtype is not None else get_stat_dtype(device)
    if ":" in spec:
        path, ref = spec.split(":", 1)
        obj = torch.load(os.path.join(checkpoint_dir, path), map_location="cpu", weights_only=False)
        for k in ref.split("."):
            obj = obj[k]
        return obj.to(dtype=target_dtype).to(device)
    obj = torch.load(os.path.join(checkpoint_dir, spec), map_location="cpu", weights_only=False)
    return obj.to(dtype=target_dtype).to(device)


def build_l0_prior(
    checkpoint_dir: str,
    spec: str | None,
    l0_mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if not spec:
        return None
    l0_p = load_allowance(checkpoint_dir, str(spec), device, dtype=torch.float32).float()
    mode = l0_mode or "unigram"
    if mode == "active":
        return (l0_p > 0).float() / (l0_p > 0).sum()
    if mode == "uniform":
        return torch.ones_like(l0_p) / l0_p.numel()
    if mode.startswith("smooth:"):
        eps = float(mode.split(":", 1)[1])
        return (1.0 - eps) * l0_p + eps * torch.ones_like(l0_p) / l0_p.numel()
    return l0_p


def parse_adj(adj_spec: str | None) -> list[float] | None:
    if not adj_spec:
        return None
    return [float(v) for v in str(adj_spec).split(",")]


def load_models(cfg, checkpoint_dir: str, device: torch.device):
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(checkpoint_dir, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(checkpoint_dir, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(os.path.join(checkpoint_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    return tk, net


def pick_tokens(
    logits: torch.Tensor,
    device: torch.device,
    freq: float = 0.0,
    mu: float = 0.0,
    allow: torch.Tensor | None = None,
    adj: list[float] | None = None,
) -> torch.Tensor:
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    B, T, V = logits.shape
    stat_dtype = get_stat_dtype(device)
    lg = logits.to(stat_dtype)
    allow_t = (allow[:V] * T)[None, :].to(stat_dtype) if allow is not None else None
    al = adj[0] if adj else 0.0
    aw = int(adj[1]) if adj and len(adj) > 1 else 1
    ad = adj[2] if adj and len(adj) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=stat_dtype, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - allow_t).clamp_min(0.0) if allow_t is not None else cnt
            step -= freq * excess
            if mu > 0 and allow_t is not None:
                step += mu * ((T - t) / T) * (allow_t - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=stat_dtype, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=stat_dtype, device=lg.device))
    return out


def decode_ids(tk, ids, eos, pad, cut_at_eos: bool = True) -> str:
    ids = list(ids)
    if cut_at_eos and eos is not None and eos in ids:
        ids = ids[:ids.index(eos)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True)


def generate_batch(
    net,
    tk,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    l0: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.0,
    cfg_scale: float = 0.0,
    eos: int | None = None,
    pad: int | None = None,
    freq: float = 0.0,
    mu: float = 0.0,
    allow: torch.Tensor | None = None,
    adj: list[float] | None = None,
    cut_at_eos: bool = True,
) -> tuple[list[str], list[list[int]]]:
    autocast_enabled = dtype in (torch.bfloat16, torch.float16)
    cfg_null_eos = eos if eos is not None else 0
    with torch.no_grad():
        with xpu_compat.autocast_context(device, dtype=dtype, enabled=autocast_enabled):
            g = net.generate(
                l0,
                x,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale,
                cfg_null_eos=cfg_null_eos,
            )
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            logits = tk.vqvae.decoder(zl[-1])
        ids = pick_tokens(logits, device=device, freq=freq, mu=mu, allow=allow, adj=adj).tolist()
    texts = [decode_ids(tk, r, eos=eos, pad=pad, cut_at_eos=cut_at_eos) for r in ids]
    return texts, ids


def resolve_data_dir(cfg, data_dir_arg: str, split: str) -> str:
    data_root = os.environ.get("NCM_DATA_ROOT", "data")
    task = str(cfg.task)
    dd = data_dir_arg or os.path.join(data_root, task)
    if not cfg.data or not cfg.data.get(split):
        return dd
    target_file = str(cfg.data[split])
    if os.path.exists(os.path.join(dd, target_file)):
        return dd
    if "-" in task:
        prefix_part, sub_part = task.split("-", 1)
        for candidate in (
            os.path.join(dd, sub_part),
            os.path.join(data_root, prefix_part, sub_part),
        ):
            if os.path.exists(os.path.join(candidate, target_file)):
                return candidate
    return dd


def main():
    a = parse_args()
    CFG = OmegaConf.load(a.config)
    G = CFG.generation
    ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
    split = a.split or CFG.eval.get("split", "test")
    dd = resolve_data_dir(CFG, a.data_dir, split)
    pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
    temp = pick(a.temp, "temperature", float, 1.0)
    top_k = pick(a.top_k, "top_k", int, 1)
    top_p = pick(a.top_p, "top_p", float, 0.0)
    cfg_scale = pick(a.cfg, "cfg_scale", float, 0.0)
    freq = pick(a.freq, "frequency_penalty", float, 0.0)
    mu = pick(a.mu, "frequency_deficit", float, 0.0)
    adj_spec = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
    n = a.n or int(CFG.eval.get("n_docs", 0))
    bs = a.bs or int(CFG.eval.batch_size)

    device = xpu_compat.resolve_device(a.device)
    xpu_compat.configure_matmul_precision(device)
    dtype = resolve_dtype(a.dtype, device)
    print(f"shard {a.shard}: device={device} ({device_display_name(device)}) dtype={str(dtype).split('.')[-1]}", flush=True)

    tk, net = load_models(CFG, ck, device)
    P = int(CFG.model.prefix_len)
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    ALLOW = load_allowance(ck, str(G.frequency_allowance), device) if G.get("frequency_allowance") else None
    L0_P = build_l0_prior(ck, G.get("level0_unigram"), a.l0 or "unigram", device)
    UNCOND = not CFG.data or not CFG.data.get(split)
    ADJ = parse_adj(adj_spec)
    CUT_AT_EOS = bool(G.get("cut_at_eos", True))

    if UNCOND:
        seeds = [int(x) for x in CFG.eval.seeds][a.shard::a.nshards]
        n = a.n or int(CFG.eval.n_samples)
        items, refs, idxs, src_tok = [], [], [], None
    else:
        with open(os.path.join(dd, CFG.data[split]), "rb") as f:
            items = pickle.load(f)[:n]
        with open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb") as f:
            refs = pickle.load(f)[:n]
        assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
        idxs = list(range(a.shard, len(items), a.nshards))
        src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if (items and isinstance(items[0], str)) else None

    pad_val = PAD if PAD is not None else (CFG.model.get("prefix_pad_token_id") or 0)

    def prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [pad_val] * P)[:P]

    RAW = []

    def run(x, l0=None):
        texts, ids = generate_batch(
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
            eos=EOS,
            pad=PAD,
            freq=freq,
            mu=mu,
            allow=ALLOW,
            adj=ADJ,
            cut_at_eos=CUT_AT_EOS,
        )
        RAW.extend(ids)
        return texts

    hyps, sample_seed = [], []
    if UNCOND:
        L0 = tk.vqvae.vq.levels[0]
        for seed in seeds:
            xpu_compat.seed_everything(seed)
            for s in range(0, n, bs):
                B = min(bs, n - s)
                l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).to(device) if L0_P is not None else None
                hyps += run(torch.full((B, 1), EOS, dtype=torch.long, device=device), l0)
                sample_seed += [seed] * B
            print(f"  shard{a.shard} seed {seed}: {n} samples", flush=True)
        idxs = list(range(len(hyps)))
        toks = [tk.pre_tokenizer.encode(h) for h in hyps]
        out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps), "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
    else:
        for s in range(0, len(idxs), bs):
            batch = idxs[s:s + bs]
            hyps += run(torch.tensor([prefix(items[i]) for i in batch], dtype=torch.long, device=device))
            if a.shard == 0 and (s // bs) % 10 == 0:
                print(f"  shard0 {s}/{len(idxs)}", flush=True)
        out = {"idxs": idxs, "hyps": hyps, "refs": [r if isinstance(r, str) else decode_ids(tk, r, EOS, PAD, CUT_AT_EOS) for r in (refs[i] for i in idxs)]}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"shard {a.shard}: {len(hyps)} -> {a.out}")


if __name__ == "__main__":
    main()

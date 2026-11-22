import argparse
import glob
import os
import pickle
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = os.environ.get("NCM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xpu_compat


def load_ref_model(ref_name: str, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref_name)
    model_dtype = torch.bfloat16 if device.type in ("cuda", "xpu") else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(ref_name, torch_dtype=model_dtype).to(device).eval()
    return tok, lm


def bleu(hyps, refs):
    import sacrebleu

    b = sacrebleu.corpus_bleu(hyps, [refs])
    bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    print(f"{'BLEU':>8} {'BLEU(lc)':>9} {'chrF':>7} {'len ratio':>10}")
    print(f"{b.score:8.2f} {bl.score:9.2f} {c.score:7.2f} {lr:10.3f}")
    print(f"\n{b.format(signature=True) if hasattr(b, 'format') else ''}")


def rouge(hyps, refs):
    import torch
    from rouge_score import rouge_scorer
    from evaluation.gen_ppl import generation_perplexity

    device = xpu_compat.resolve_device()
    xpu_compat.configure_matmul_precision(device)
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    g2, lm = load_ref_model("gpt2", device)

    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in R:
            R[k].append(v[k].fmeasure)
    with torch.no_grad():
        gp = generation_perplexity(
            lm, g2, [""] * len(hyps), hyps,
            device=str(device), score_only_generated=True, batch_size=32,
        ).mean_ppl
    print(f"{'R-1':>7} {'R-2':>7} {'R-L':>7} {'genPPL':>8} {'words':>7}")
    print(f"{100 * np.mean(R['rouge1']):7.2f} {100 * np.mean(R['rouge2']):7.2f} {100 * np.mean(R['rougeL']):7.2f} "
          f"{gp:8.2f} {np.mean([len(h.split()) for h in hyps]):7.1f}")


def exact(hyps, refs):
    def cells(t):
        return t.replace("|", "")

    def valid(t):
        g = cells(t)
        if len(g) != 81 or not g.isdigit() or "0" in g:
            return False
        rows_ = [g[i * 9:(i + 1) * 9] for i in range(9)]
        cols_ = ["".join(r[j] for r in rows_) for j in range(9)]
        boxes = ["".join(rows_[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)) for br in (0, 3, 6) for bc in (0, 3, 6)]
        return all(sorted(x) == list("123456789") for x in rows_ + cols_ + boxes)

    n = len(hyps)
    exact_n = sum(h == r for h, r in zip(hyps, refs))
    cell = np.mean([np.mean([a == b for a, b in zip(cells(h), cells(r))]) for h, r in zip(hyps, refs)])
    valid_n = sum(valid(h) for h in hyps)
    print(f"{'solved %':>9} {'solved':>7} {'cell %':>7} {'valid %':>8}")
    print(f"{100 * exact_n / n:9.2f} {exact_n:4d}/{n} {100 * cell:7.2f} {100 * valid_n / n:8.2f}")
    return {
        "solved_pct": 100 * exact_n / n,
        "solved": exact_n,
        "total": n,
        "cell_pct": 100 * float(cell),
        "valid_pct": 100 * valid_n / n,
    }


def genppl(cfg, seeds, hyps_by_part, toks):
    import math
    import torch
    from collections import Counter
    from evaluation.gen_ppl import generation_perplexity

    device = xpu_compat.resolve_device()
    xpu_compat.configure_matmul_precision(device)
    ref = str(cfg.eval.ref_model)
    rtok, rlm = load_ref_model(ref, device)

    def entropy(ids):
        c, m = Counter(ids), len(ids)
        return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0

    per = []
    for seed in dict.fromkeys(seeds):
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        gen = [hyps_by_part[k] for k in sel]
        with torch.no_grad():
            ppl = generation_perplexity(rlm, rtok, [""] * len(gen), gen, device=str(device), batch_size=32).mean_ppl
        ent = float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))
        per.append((seed, ppl, ent))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per])
    e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(hyps_by_part)} samples)")


def genppl_flm(cfg, seeds, raw_ids):
    # FLM protocol: decode with specials kept, rescore the first 128 GPT-2 tokens, corpus-level perplexity
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    device = xpu_compat.resolve_device()
    xpu_compat.configure_matmul_precision(device)
    gen_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
    ref = str(cfg.eval.ref_model)
    rtok, rlm = load_ref_model(ref, device)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            for i in range(0, ids.shape[0], 16):
                s_, m_ = ids[i:i + 16], attn[i:i + 16]
                logits = rlm(s_, attention_mask=m_)[0].float().transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum())
                tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    def entropy(ids):
        t = torch.tensor(ids)
        return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

    per = []
    for seed in dict.fromkeys(seeds):
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        texts = gen_tok.batch_decode([raw_ids[k] for k in sel])
        per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw_ids[k]) for k in sel]))))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per])
    e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(raw_ids)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--glob", required=True)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    parts = []
    for path in sorted(glob.glob(a.glob)):
        with open(path, "rb") as f:
            parts.append(pickle.load(f))
    assert parts, "no shard files"
    rows = sorted((i, h, r) for p in parts for i, h, r in zip(p["idxs"], p["hyps"], p["refs"]))
    seeds = [s for p in parts for s in p.get("seeds", [])]
    toks = [t for p in parts for t in p.get("toks", [])]
    raw_ids = [r for p in parts for r in p.get("raw_ids", [])]
    hyps_by_part = [h for p in parts for h in p["hyps"]]
    hyps = [h for _, h, _ in rows]
    refs = [r for _, _, r in rows]
    print(f"\nmerged {len(parts)} shards -> {len(rows)} {cfg.task} outputs\n")

    metric = str(cfg.eval.metric)
    dispatch = {
        "bleu": lambda: bleu(hyps, refs),
        "rouge": lambda: rouge(hyps, refs),
        "exact": lambda: exact(hyps, refs),
        "genppl": lambda: genppl(cfg, seeds, hyps_by_part, toks),
        "genppl_flm": lambda: genppl_flm(cfg, seeds, raw_ids),
    }
    dispatch[metric]()


if __name__ == "__main__":
    main()

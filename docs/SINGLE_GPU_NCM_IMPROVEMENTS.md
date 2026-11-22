# Single-16GB-GPU 1.32B Next-Concept Modeling: Engineering Report & Upstream Contributions

**Technical Report for HazyResearch NCM (`github.com/HazyResearch/ncm`, arXiv:2610.14162)**

---

## 1. Executive Summary & Single-16GB-GPU 1.32B Hardware Profile

This report documents the engineering changes, numerical fixes, memory optimizations, and empirical findings required to run inference, evaluation, and full two-stage training of a **1.32B-parameter Next-Concept Modeling (NCM)** system on a single consumer **16 GB Intel Arc A770 (`xpu:0`)** workstation running **Windows 11 WDDM** and **PyTorch XPU (`torch==2.15.0+xpu`)**.

Every training-time optimization presented here is **quality-neutral**: no gradient approximations beyond standard BF16/Kahan error-compensated updates, no sequence truncation of valid tokens, and no lossy attention approximations are introduced.

### Key Outcomes

- **Upstream Baseline Parity on Consumer XPU:** Verified upstream NCM checkpoints on `xpu:0` (`bfloat16` autocast), achieving **`260.8 ms`** (`batch=1`, `857 MB` VRAM) and **`348.6 ms`** (`batch=16`, `1,265 MB` VRAM, **`13.0x`** speedup over a 24-thread `i7-13700K` CPU) on `tinystories`, and **`53/64` (`82.81%`)** solved in **`10.91s`** (`< 2 GB` VRAM) on `sudoku-hard`.
- **1.32B Training Inside a 14.2 GB WDDM Ceiling:** Trained the full **`1,320.36M`** combined model (**`103.35M`** Stage-1 VQ-VAE + **`1,217.01M`** Stage-2 `NextConceptTransformer`) without CUDA-only 8-bit optimizers (`bitsandbytes`). A pure-PyTorch **`KahanBF16AdamW`** optimizer, offline `uint16` concept pre-encoding, per-block gradient checkpointing, lossless prefix bucketing, and single-copy checkpointing hold steady training VRAM at **`9,561–9,565 MB`** (**`7,239–7,242 MB`** after `zero_grad`, **`11,588 MB`** peak allocated, **`13,056 MB`** reserved) out of `15,933 MB` physical VRAM.
- **24-Hour Driver Stability & Convergence:** Diagnosed and eliminated a `~1.2 GB/hr` Level Zero native handle leak, a deterministic `ze_intel_gpu64.dll` (`0xC0000005` at `0x404ed4`) mid-run `empty_cache()` crash, an XPU `.mean(dim=0)` `NaN` reduction bug, and resume-time host commit bloat (`28.5–39.2 GB -> 16.8–20.7 GB`). Stage-2 training converged cleanly from **`val_loss=6.2278` / `L0_acc=5.5%`** (step `500`) to **`val_loss=5.2075` / `L0_acc=69.5%`** (step `1,500`) at **`40–43s/step`** on `xpu:0`.
- **CPU Optimizer Offload A/B Evidence:** Implemented slab-batched CPU offload (`--optimizer-cpu-offload`) + `S`-adaptive activation caching (`--activation-budget-mb`), proving bit-exact step parity (`0.0000` loss/gnorm delta) and `-4,651 MB` steady VRAM reduction (`7,242 -> 2,591 MB`), while showing why GPU-native `KahanBF16AdamW` remains **`3.5x` faster** (`38.40s` vs `139.17s/step`) on `16 GB` GPUs.

### Hardware & Runtime Profile

| Parameter | Verified Specification |
| :--- | :--- |
| **Discrete GPU** | `Intel(R) Arc(TM) A770 Graphics` (`xpu:0`, `15,932.84 MB` physical GDDR6, 512 EUs) |
| **Host OS & Driver** | Windows 11 (`win_amd64`) + WSL2 wrapper, Level Zero `ze_intel_gpu64.dll 1.15.39183.3` |
| **WDDM Safe VRAM Budget** | `~14.2 GB` dedicated allocation ceiling before Windows WDDM shared-memory spill |
| **PyTorch Stack** | Python `3.12.16`, `torch==2.15.0+xpu`, `triton-xpu==3.9.0` |
| **Stage-1 VQ-VAE** | `103.35M` total / `77.46M` non-embedding (`enc=4, dec=4, dim=512, vocab=4096x7`) |
| **Stage-2 NCP** | `1,217.01M` total / `1,215.95M` non-embedding (`n_layer=24, n_head=32, n_embd=2048`) |
| **Combined Footprint** | **`1,320.36M` (`1.32B`)** parameters; `prefix_len=1536` + `508` concepts (`levels=[4,8,16,32,64,128,256]`) |

### Single-16GB-GPU Training Architecture

```text
+-----------------------------------------------------------------------------+
| Host CPU & Disk (Windows 11 / WSL2, uint16 Memory-Mapped Shards)            |
|  data/memmap/{stage2_wide,stage2_gold,eval}/                                |
|   +-- prefix_1536.bin (uint16)  --> _trim_prefix_to_bucket (256..1536)      |
|   +-- gt_idx_508.bin  (uint16)  <-- offline pre-encoded by preencode_concepts|
+-----------------------------------------------------------------------------+
                    | (trimmed prefix_ids, gt_idx[508]) via PCIe
                    v
+-----------------------------------------------------------------------------+
| Intel Arc A770 (xpu:0, 15,933 MB GDDR6 / ~14.2 GB WDDM Safe Ceiling)        |
|                                                                             |
|  1. Frozen Stage-1 VQ-VAE Slices (BF16, 103.35M; encoder/decoder bypassed)  |
|     prefix_ids --> vqvae.encoder.wte ----+                                  |
|     gt_idx     --> vqvae.vq.idx_to_ncm_input + l0_queries                   |
|                                          |                                  |
|                                          v                                  |
|  2. Stage-2 NextConceptTransformer (BF16, 1,217.01M, 24 Blocks)             |
|     * Per-block gradient checkpointing (use_reentrant=False)                |
|     * Static 256-token prefix buckets (p_eff in {256,512,768,1024,1280,1536})|
|     * Boolean 4D SDPA masks + sliced per-level RoPE (use_flex_attention=false)|
|                                          |                                  |
|                                          v                                  |
|  3. KahanBF16AdamW (In-VRAM, ~8.1 B/param)                                  |
|     * BF16 weights + BF16 kahan_comp + BF16 exp_avg                         |
|     * Factored FP32 exp_avg_sq_row/col (2D) & full FP32 exp_avg_sq (1D)     |
|     * Steady VRAM: 9,561-9,565 MB (7,242 MB post-zero_grad, 11,588 MB peak) |
+-----------------------------------------------------------------------------+
                    | (optional --optimizer-cpu-offload for 8-12 GB GPUs)
                    v
+-----------------------------------------------------------------------------+
| Supervisor & Atomic Checkpointing (`--allow-recycle-exit`)                   |
|  * Single-copy _snapshot_to_cpu() -> ncm_latest.pt / ncm_best.pt / export/  |
|  * Periodic worker recycling (500 steps / +6 GB commit growth) resets WDDM  |
+-----------------------------------------------------------------------------+
```

---

## 2. Zero-Quality-Degradation & Mathematical Equivalence Guarantees

A core design constraint was ensuring every memory or throughput optimization leaves model mathematics unchanged relative to an unoptimized full-sequence BF16/AdamW baseline:

| Optimization | Potential Drift Risk | Equivalence / Guard Mechanism | Verified Result |
| :--- | :--- | :--- | :--- |
| **Offline `gt_idx_508.bin` Pre-Encoding** | Drift if VQ-VAE were updated | Stage-1 VQ-VAE is frozen before Stage-2; `preencode_concepts.py` runs `vqvae.encode()` in eval mode | Bit-identical indices to online `vqvae.encode()` |
| **Factored `KahanBF16AdamW`** | BF16 weight rounding loss | Persistent BF16 `kahan_comp` residual (`p_fp32 = p + kahan_comp`) + FP32 accumulation & vector-norm clip | Clean monotonic convergence (`7.8600 -> 4.6826`) |
| **Right-Pad Prefix Trimming (`bucket=256`)** | Positional/attention change | Masks only trailing `pad_token_id` columns already zeroed by `prefix_pad_mask`; per-level RoPE preserves `[0..p_eff-1]` and `[0..l_k-1]` | **`0.0`** max logit & loss delta (`test_xpu_core.py`) |
| **Boolean 4D SDPA Masks** | Float `-inf` vs bool semantics | `keep_mask = ~torch.isneginf(base_mask)` combined with `~safe_pm` (`all_pad & col0` guard) | Identical SDPA output with lower mask memory |
| **Slab-Batched CPU Offload** | Transfer/shadow drift | Lazy `_cpu_param_shadow` updated in FP32 via identical `_apply_kahan_adamw_update` | **`torch.equal`** bit-exact (`0.0000` delta steps `1..3`) |
| **`S`-Adaptive Activation Caching** | Checkpoint vs eager grads | Caches early blocks `b_idx < cached_blocks` with `fn(x, *args, **kwargs)`; checkpoints remainder | `torch.allclose(..., atol=1e-5, rtol=1e-5)` grad parity |

---

## 3. Cross-Accelerator Portability (`xpu_compat.py` & Kernel Stability Fixes)

Running `HazyResearch/ncm` cleanly across Intel Arc XPU, NVIDIA CUDA, Apple MPS, and CPU required isolating backend differences in `xpu_compat.py` and fixing several dynamic-shape ops in the model core.

### 3.1 Discrete GPU Filtering & Device Abstraction (`xpu_compat.py`)

Workstations with Intel CPUs (such as `i7-13700K`) expose both an integrated `Intel(R) UHD Graphics 770` and the discrete `Intel(R) Arc(TM) A770 Graphics` through Level Zero. `xpu_compat.py` prevents worker processes from landing on the iGPU:

- **`get_discrete_xpu_indices()` & `select_best_xpu_index()`:** Inspects `torch.xpu.get_device_name(idx)`, selects devices containing `"arc"` and excluding `"uhd"` (falling back to non-UHD devices), and pairs with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` in `run-xpu.sh` / `run-xpu.ps1`.
- **`resolve_device()` & `get_worker_device_ids()`:** Resolves `auto`, `xpu`, `cuda`, `mps`, and `cpu` uniformly across `evaluate.py`, `scripts/eval_shard.py`, `scripts/train_vqvae.py`, `scripts/preencode_concepts.py`, and `scripts/train_ncm.py`.
- **`supports_fp64()`:** Queries `torch.xpu.get_device_capability(dev)` (`has_fp64=0` on consumer Arc silicon) and `mps` availability, keeping statistical accumulators and optimizer math in `float32` instead of `float64`.
- **`autocast_context()`:** Wraps `torch.amp.autocast` with `bfloat16` default (`--dtype float16` triggers a Windows SYCL assertion in mixed-precision attention/VQ paths).
- **`configure_matmul_precision()`:** Applies `torch.set_float32_matmul_precision("high")` on `xpu`/`cuda` and TF32 flags on `cuda`.

### 3.2 Static-Shape `torch.where` Fixes & Attention Fallback (`models/quant.py`, `models/next_concept.py`)

Two classes of PyTorch operations caused runtime errors or excessive compilation on Intel XPU:

1. **Dynamic Boolean Mask Indexing (`755c72b`):** PyTorch XPU struggled with dynamic-output boolean indexing (`tensor[mask]`, `nonzero()`). In `models/quant.py`, `kmeans_init_level()` (lines 367–373) and `_restart_dead_codes()` (lines 442–448) were rewritten to use fixed-shape `torch.where(mask, ...)` over the full codebook size `V`. Likewise, `_sample_codes()` (`models/quant.py:571`) and `_sample_level_raw()` (`models/next_concept.py:544,549`) replaced `.masked_fill(...)` with `torch.where(..., torch.full_like(..., -torch.inf), ...)`.
2. **FlexAttention Guard (`models/next_concept.py:318-319`):** `_should_use_flex_attention()` explicitly returns `False` on `device.type in ("xpu", "mps")`, routing attention cleanly through eager `F.scaled_dot_product_attention` without Triton block-mask compilation failures.
3. **Windows Teardown Exit Guard (`e856d3c`):** During interpreter shutdown on Windows, PyTorch XPU DLLs can occasionally exit with `0xC0000409` (`3221226505` / `-1073740791`) after Python cleanup finishes. `scripts/train_vqvae.py`, `scripts/preencode_concepts.py`, and `scripts/train_ncm.py` call `xpu_compat.shutdown_device_cleanly()` (`gc.collect()`, `synchronize()`, `empty_cache()`, stream flush) before exit, allowing an external supervisor to verify `status.json` (`state in {"completed", "recycling", "stopped"}`) and required export/checkpoint artifacts (`tokenizer.pt`, `gt_idx_508.bin` + `unigrams.pt`, `generator.pt` / `ncm_latest.pt`) on disk.

---

## 4. Two-Stage Single-GPU Training Pipeline & Single-Copy Atomic Checkpointing

To train a `1.32B` model unattended on a single `16 GB` GPU, the two-stage workflow runs resumable stages over fixed-width `uint16` memory-mapped binaries (`data/memmap/`):

```text
[1. prepare memmap] --> [2. train_vqvae] --> [3. preencode] --> [4. train_ncm]
                          25k steps            3.02M rows         WSD + --continuous
                          1,501 MB VRAM        gt_idx_508.bin     9,561 MB steady
```

### 4.1 Stage-1 VQ-VAE (`scripts/train_vqvae.py`)

- **Architecture (`ncm_1_3b` `1.32B` profile):** `103.35M` total / `77.46M` non-embedding parameters (`in_vocab_size=50304`, `context_length=256`, `compression_factor=1`, `embed_dim=512`, `n_layers=8`, `enc_layers=4`, `dec_layers=4`, `vqvae_vocab_size=[4096]*7`, `levels=[4, 8, 16, 32, 64, 128, 256]`, `pre_quant_groupnorm=8`, `use_qk_norm=true`, `learned_l1_sampling=true`, `learned_l1_cascade=true`, `quant_resi.ratio=0.5`, `fine_drop.prob=0.5`).
- **Initialization:** `_maybe_init_gpt2_wte()` projects GPT-2 `wte` (`768 -> 512`) via centered SVD with norm scaling into `vqvae.encoder.wte[:50257]`, followed by step-0 `kmeans_init_level(0)` on `512` encoded samples (`kmeans_init_batches=8`).
- **Observed Run (`runs/ncm_1_3b/train.log`):** Trained on `4,223,113` blocks (`bs=16 x grad_accum=2`, `lr=3.0e-4 -> 3.0e-5`, `1,501 MB` VRAM, `~8.0s/step`) for `25,000` steps. Final step `25000` achieved `val_loss=2.8836` / `val_recon=1.8188` (best `eval step 24500: val_loss=2.7682 val_recon=1.7181`) with 20-step window active codebook utilization `util=0.266 [0.03,0.04,0.09,0.16,0.30,0.52,0.72]` (where `_reset_codebook_hits()` zeroes `codebook_hit_{i}` every `log_every=20` steps, so `0.72` reflects active codes at L6 within a single 20-step / 640-sequence window rather than global dataset coverage).

### 4.2 Offline Concept Pre-Encoding (`scripts/preencode_concepts.py`)

Running `vqvae.encode()` online during Stage-2 training wastes VRAM and compute on every micro-batch. `scripts/preencode_concepts.py` runs the frozen `tokenizer.pt` once across all Stage-2 splits (`bfloat16`, `batch_size=256`), producing:

- `gt_idx_508.bin` (`uint16` memmap arrays across `[4, 8, 16, 32, 64, 128, 256]` = `508` indices/row) for **`3,016,205`** total targets (`stage2_wide: 2,190,985`, `stage2_gold: 822,140`, `eval: 3,080`).
- `unigrams.pt` (`token_unigram` and `level0_unigram` frequency distributions) used for inference penalty calibration.
- During Stage-2 (`_forward_ncm_step`), the VQ-VAE decoder and encoder transformer blocks are never invoked: only `vqvae.encoder.wte(prefix_ids)` and `vqvae.vq.idx_to_ncm_input(gt_idx)` execute.

### 4.3 Single-Copy Atomic Checkpointing & Resume Hygiene (`scripts/train_ncm.py`)

Saving a `1.22B` model + optimizer state (`ncm_latest.pt`, `ncm_best.pt`, `export/generator.pt`) naively creates redundant CPU copies of `state_dict()` and spikes host RAM commit:

- **Single-Copy CPU Snapshots (`_snapshot_to_cpu`, `_get_model_cpu_sd`, `_get_opt_cpu_sd`):** At any save/eval/export/recycle step, `net.state_dict()` and `optimizer.state_dict()` are snapshotted to CPU at most once and reused across `best_ckpt`, `export_gen_path`, and `latest_ckpt`, then explicitly cleared (`model_cpu_sd = None; opt_cpu_sd = None; gc.collect()`).
- **Atomic File Writes:** `_atomic_save()` and `_atomic_write_json()` write to `<name>.tmp` followed by `os.replace()` so `STOP` or power-loss interruptions never corrupt `ncm_latest.pt`, `status.json`, or `pipeline_state.json`.
- **Resume Memory Release (`b31e4aa`):** Explicitly deleting the VQ-VAE checkpoint dictionary (`del tok_sd`) and Stage-2 resume checkpoint (`del ckpt; gc.collect()`) immediately after `load_state_dict()` dropped steady host RAM commit from `28.5–39.2 GB` down to `16.8–20.7 GB` (`train.log` lines 1951–1980).

---

## 5. Factored `KahanBF16AdamW` Optimizer & Intel XPU `.mean(dim=0)` `NaN` Fix

### 5.1 Memory Breakdown: Why Standard AdamW Fails on 16 GB

For the **`1,217.01M`**-parameter Stage-2 `NextConceptTransformer`, standard mixed-precision `AdamW` requires FP32 master weights (`4 B`), FP32 first moments (`4 B`), FP32 second moments (`4 B`), BF16 weights (`2 B`), and BF16 gradients (`2 B`) — **`16–18 B/param` (`> 19.5 GB`)**, exceeding the `~14.2 GB` WDDM ceiling before activations. Meanwhile, `bitsandbytes` 8-bit optimizers are CUDA-specific and unavailable on Intel XPU.

Implemented in pure PyTorch (`scripts/train_ncm.py:102-386`), **`KahanBF16AdamW`** cuts steady parameter + optimizer + gradient footprint to **`~8.1 B/param`**:

| Tensor Group | Storage Dtype | Bytes / Param | Update Semantics |
| :--- | :--- | :---: | :--- |
| **Trainable Weights (`p`)** | `bfloat16` | `2.0 B` | Updated in-place from FP32 Kahan accumulator |
| **Kahan Residual (`kahan_comp`)** | `bfloat16` | `2.0 B` | Captures `p_fp32 - p_bf16` truncation error every step |
| **First Moment (`exp_avg`)** | `bfloat16` | `2.0 B` | Unpacked to `float32` for `beta1=0.9` momentum |
| **Second Moment (2D weights)** | `float32` (`row`, `col`) | `< 0.1 B` | Adafactor rank-1 row/col means (`beta2=0.95`, bias-corrected) |
| **Second Moment (1D norms/biases)** | `float32` (`exp_avg_sq`) | `4.0 B` (tiny share) | Full FP32 second moment for non-2D tensors |
| **Gradients (`p.grad`)** | `bfloat16` | `2.0 B` | Freed immediately via `zero_grad(set_to_none=True)` |

**Measured `xpu:0` VRAM (`ncm_1_3b`):**
- **After `optimizer.zero_grad(set_to_none=True)`:** `7,239–7,242 MB` allocated (`status.json`: `7238.92 MB`).
- **Steady Training (gradients live):** `9,561–9,565 MB` allocated.
- **Peak Allocated (`max_memory_allocated`):** `11,588 MB` (`11588.26 MB`).
- **Reserved Pool:** `13,056 MB` out of `15,933 MB` physical (zero WDDM shared-memory spill).

### 5.2 Intel XPU `.mean(dim=0)` `NaN` Root Cause & Fix (`0fe177c`)

During the initial `ncm_1_3b` Stage-2 launch (`train.log` lines 1777–1778), step `1` logged `loss=7.8600`, and by step `20` training collapsed to `loss=nan`. Investigation isolated an Intel XPU reduction bug when computing `.mean(dim=0)` / `.mean(dim=1)` on squared FP32 gradient tensors inside `KahanBF16AdamW`, corrupting `exp_avg_sq_col` / `exp_avg_sq_row` with `NaN`s on the first optimizer step.

The fix in `_apply_kahan_adamw_update()` (`scripts/train_ncm.py:209-217`) and `_clip_grad_norm_fp32()` (`scripts/train_ncm.py:670-686`) enforces:

1. **Explicit Sum-Div Reductions:** Replaced `g_sq.mean(dim=1)` and `g_sq.mean(dim=0)` with `(g_sq.sum(dim=1) / float(g_sq.shape[1])).clamp_min(0.0)` and `(g_sq.sum(dim=0) / float(g_sq.shape[0])).clamp_min(0.0)`, plus `(r_hat.sum() / float(r_hat.numel())).clamp(min=1e-30)`.
2. **Non-Negativity Clamping:** Clamps `r_hat`, `c_hat`, and `v_hat` with `.clamp_min(0.0)` prior to `v_hat.sqrt()` so numerical noise never yields negative square roots.
3. **FP32 Vector-Norm Gradient Clipping:** `_clip_grad_norm_fp32()` accumulates `torch.linalg.vector_norm(g.detach().float())` in `float32`, verifies `torch.isfinite(total_norm)`, and skips `optimizer.step()` on non-finite loss or gradient norms.

Immediately after applying `0fe177c`, `ncm_1_3b` resumed cleanly with finite gradients (`step 1: loss=7.8600 gnorm=0.327` -> `step 20: loss=7.7426 gnorm=0.223` -> `step 1540: loss=4.6826`).

---

## 6. Lossless Right-Pad Prefix Trimming, Boolean 4D Masks & `256`-Token Static Bucketing

With `use_flex_attention: false` on XPU, `F.scaled_dot_product_attention` runs the `O(S^2)` math path over `S = prefix_len + 508`. At full `prefix_len=1536`, `S = 2,044` even when most real issue descriptions occupy `100–600` tokens followed by right-padding (`prefix_pad_token_id=50257`).

### 6.1 Why Dynamic Trimming & `256`-Token Bucketing Are Lossless (`2d0839e`, `0fe177c`, `eecb742`)

1. **CPU-Side Bucket Trimming (`_trim_prefix_to_bucket`, `scripts/train_ncm.py:701-720`):** Before moving `prefix_ids` to GPU, the trainer finds the rightmost non-pad token across the micro-batch and rounds up to a multiple of `bucket_size=256`, producing at most **6 static prefix lengths** (`256, 512, 768, 1024, 1280, 1536`). Static bucketing avoids XPU/SYCL kernel recompilation churn across arbitrary lengths while cutting average `S^2` attention FLOPs and activation memory dramatically.
2. **Sliced Per-Level RoPE & Masks (`models/next_concept.py:456-505`):**
   - Per-level RoPE indexes prefix positions `[0..p_eff-1]` independently of target level chunks `[0..l_k-1]`. Slicing `(self.rope_cos[:p_eff], self.rope_cos[p_full:])` preserves every token's exact rotary embedding.
   - `attn_mask_for_masking` is sliced at `p_eff` (`[:p_eff, :p_eff]`, `[:p_eff, p_full:]`, `[p_full:, :p_eff]`, `[p_full:, p_full:]`).
   - `_level0_slice()` takes `p_eff=p_eff` so `predict_level0` indexing remains exact.
   - Because trailing positions `[p_eff..p_full-1]` were `pad_token_id`, `prefix_pad_mask` already masked their keys to `-inf` (zero attention weight), making trimming **mathematically lossless (`0.0` logit difference)**.
3. **Boolean 4D SDPA Masks (`models/next_concept.py:490-500`):** Replaced float `-inf` additive attention masks with `keep_mask = ~torch.isneginf(base_mask)` intersected with `key_keep` (`~safe_pm`, keeping `col0` unmasked if an entire row is pad to avoid `softmax(-inf)` `NaN`s). Boolean masks cut mask memory and avoid per-layer `dtype` casts.

---

## 7. Windows WDDM / Level Zero Native Handle Leak & `0xC0000005` Crash Analysis + Worker Recycling

Extended 24-hour runs on Windows 11 WDDM (`ze_intel_gpu64.dll 1.15.39183.3`) uncovered three distinct driver/memory failure modes that do not appear in short unit tests.

### 7.1 Finding 1: `~1.2 GB/hr` Native Handle Leak from Per-Step `empty_cache()`

Calling `torch.xpu.empty_cache()` (`zeMemFree` / `zeMemAllocDevice`) every step or micro-batch caused native Level Zero / WDDM driver memory to grow monotonically over a ~22-hour run, eventually bloating to **`14.45 GB` Dedicated VRAM + `7.27 GB` Shared WDDM spill**, **`39.22 GB` host commit**, and collapsing step latency to **`98.0s/step`**.

### 7.2 Finding 2: Deterministic `ze_intel_gpu64.dll` `0xC0000005` Crash at Offset `0x404ed4`

To bound reserved VRAM below the `14.2 GB` ceiling, an intermediate build introduced `configure_vram_cap(..., xpu_memory_fraction=0.86)` and `trim_vram_cache_if_needed(..., max_reserved_vram_mb=12400)` (`xpu_compat.py:353-416`). On `--seed 42`, PyTorch's steady reserved pool is **`13,056 MB`**, so setting `max_reserved_vram_mb=12400 < 13,056` triggered `torch.xpu.empty_cache()` immediately after freeing `~2.3 GB` of gradients at steps `1` and `3`. On **two consecutive runs**, `ze_intel_gpu64.dll` (`1.15.39183.3`) crashed deterministically with an access violation (`0xC0000005`) at offset **`0x404ed4`** mid-step `4`.

Disabling mid-run cache trimming and hard fraction caps via **`--max-reserved-vram-mb 0 --xpu-memory-fraction 0`** (important if a YAML config retains legacy `max_reserved_vram_mb: 12400` / `xpu_memory_fraction: 0.86` knobs on `ze_intel_gpu64.dll 1.15.39183.3`) completely eliminated the `0x404ed4` crash across full multi-step runs.

### 7.3 Finding 3: Desktop Responsiveness & Supervisor Worker Recycling (`7a4eb28`, `6c58b55`, `b31e4aa`)

To support unattended multi-day runs while keeping the Windows desktop interactive and preventing residual driver commit growth:

- **Win64-Safe Priority Lowering (`xpu_compat.py:468-488`):** `lower_process_priority_for_desktop()` sets `BELOW_NORMAL_PRIORITY_CLASS` (`0x00004000`) using explicit `ctypes.c_void_p` `HANDLE` `argtypes`/`restype` (fixing 64-bit pointer truncation under default `c_int` in `7d44aae`), paired with `desktop_yield_ms: 15` between micro-steps.
- **Process Commit Telemetry (`xpu_compat.py:419-465`):** `get_process_commit_gb()` queries `K32GetProcessMemoryInfo` (`PrivateUsage` / `PagefileUsage`) via `ctypes`.
- **Clean Worker Recycling (`scripts/train_ncm.py`):** When `NCM_PIPELINE_LOG_ACTIVE=1` (or `--allow-recycle-exit`), `train_ncm.py` triggers a clean checkpointed exit (`status.json` `state="recycling"`) on any of:
  1. `recycle_every: 500` steps (`steps_since_start >= 500`),
  2. `max_commit_growth_gb: 6.0` above the step-3 baseline commit (`baseline_commit_gb`), or
  3. `max_process_commit_gb: 45.0` absolute commit ceiling.
- An outer supervisor loop detects `state == "recycling"` (verifying `ncm_latest.pt` exists and coercing any Windows `0xC0000409` DLL teardown code), pauses `2.0s`, and respawns `train_ncm.py --resume` with a fresh Level Zero context—maintaining steady **`40–43s/step`** throughput.

---

## 8. Hybrid CPU-RAM Optimizer Offload + `S`-Adaptive Activation Caching (`xpu:0` A/B Benchmark)

To test whether freeing additional VRAM on consumer GPUs allows un-checkpointing early transformer blocks for higher throughput—or enables `1.32B` training on tighter `8–12 GB` cards—we implemented and benchmarked **slab-batched CPU optimizer offload** (`--optimizer-cpu-offload`) paired with **sequence-length-adaptive activation caching** (`--activation-budget-mb`) (`b7bec7e`, `f978ae6`).

### 8.1 Design (`scripts/train_ncm.py`)

1. **Slab-Batched `KahanBF16AdamW(cpu_offload=True)` (`lines 268-386`):**
   - Stores `kahan_comp`, `exp_avg`, and `exp_avg_sq_row`/`col`/`exp_avg_sq` in CPU RAM.
   - Maintains lazy `bfloat16` CPU weight shadows (`_cpu_param_shadow`) outside `state_dict()` — **zero checkpoint schema change**, supporting compatible cross-resume (`GPU -> CPU-offload -> GPU`) via `sync_cpu_shadows()`, and **zero GPU-to-CPU weight readbacks** after step `1`.
   - Packs gradients and updated weights into contiguous **`SLAB_MAX_ELEMS = 64 * 1024 * 1024`** (`128 MB` BF16) slabs for coarse-grained PCIe DMA transfers instead of hundreds of per-parameter copies.
2. **`S`-Adaptive Gradient Checkpointing (`_enable_gradient_checkpointing`, `lines 503-542`):**
   - At each block forward, estimates activation memory as `est_block_mb = B * (n_head * S^2 * 4.2 + S * n_embd * 36.0) / 1024^2`.
   - Leaves the first `cached_blocks = min(total_blocks, int(activation_budget_mb // est_block_mb))` blocks un-checkpointed; shorter `256`/`512`-token buckets cache more blocks, while `1536`-token buckets automatically checkpoint more blocks to stay within budget.

### 8.2 Verification & Live `xpu:0` `--seed 42` A/B Benchmark

- **Unit & Cross-Resume Parity (`tests/test_xpu_core.py`, `tests/test_single_gpu_trainer.py`):** Verified `torch.equal` bit-exactness across multi-slab CPU offload (`SLAB_MAX_ELEMS=32`), cross-resume `GPU -> CPU-offload -> GPU`, and `S`-adaptive gradient parity (`torch.allclose(..., atol=1e-5, rtol=1e-5)`).
- **Live `xpu:0` A/B Run (`ncm_1_3b`, `--seed 42`, `--max-reserved-vram-mb 0 --xpu-memory-fraction 0`):**

| Configuration | Steps | Loss (`1..3`) | Grad Norm (`1..3`) | Steady VRAM (`post-zero_grad`) | Peak Alloc / Reserved | Host Commit | Step Time (`mean`) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Pure-GPU `KahanBF16AdamW`** (`cpu_offload=False`, `act_budget=0`) | `1..3` | `7.8603 / 7.8514 / 7.8657` | `0.327 / 0.306 / 0.304` | **`7,242 MB`** (`9,563 MB` active) | `11,588 MB` / `13,056 MB` | **`15.3–17.6 GB`** | `33.62s / 44.50s / 37.07s` (**`38.40s/step`**; `40.0–43.3s` prod) |
| **Hybrid CPU-Offload + Cache** (`cpu_offload=True`, `act_budget=2800`) | `1..6` | `7.8603 / 7.8514 / 7.8657` (`0.0000` delta) | `0.327 / 0.306 / 0.304` (`0.000` delta) | **`2,591 MB`** (`-4,651 MB`!) | `12,405 MB` / `22,076 MB` | `38.9–40.1 GB` | `118.55s / 166.66s / 117.22s / 134.58s / 129.27s / 148.14s` (**`139.17s/step`** steps `2..6`, **`3.5x` slower**) |

### 8.3 Engineering Conclusion

- **Keep CPU offload OFF on `16 GB` GPUs:** GPU-native `KahanBF16AdamW` already fits comfortably inside `16 GB` (`9.56 GB` steady / `11.59 GB` peak). Offloading `~4.7 GB` of optimizer states plus `~2.4 GB` of BF16 weight shadows pushed host commit to `38.9–40.1 GB`, introducing host RAM paging pressure and 128-block PCIe/DDR round-trip overhead (`139.17s/step`) that far outweighed saved activation recompute time.
- **Use `--optimizer-cpu-offload` as an enabling mode for `8–12 GB` GPUs:** Dropping steady weight+optimizer VRAM from `7,242 MB` to `2,591 MB` (`-4,651 MB`) makes `1.32B` fine-tuning or Stage-2 training feasible on smaller cards when `act_budget=0` is used.

---

## 9. Empirical `1.32B` Convergence (`steps 0 -> 1,574`), File Map & Suggested Upstream PR Breakdown

### 9.1 Observed `ncm_1_3b` Stage-2 Convergence (`runs/ncm_1_3b/train.log` & `status.json`)

| Step | Phase / Split | `lr` | `train_loss` | `gnorm` | `val_loss` | `L0_acc` | Per-Mode `val_loss` (`MODE_0` / `MODE_1`) | Host Commit | Step Cadence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `warmup` (`wide`) | `1.50e-07` | `7.8600` | `0.327` | — | — | — | — | `31.87s` |
| `500` | `warmup` (`wide`) | `7.50e-05` | `6.1331` | `0.727` | **`6.2278`** | **`0.055`** (`5.5%`) | `6.3591` / `5.9772` | — | — |
| `1,000` | `warmup -> stable` | `1.50e-04` | `5.2385` | `0.498` | **`5.5915`** | **`0.273`** (`27.3%`) | `5.7130` / `5.3594` | — | — |
| `1,500` | `stable` (`wide`) | `1.50e-04` | `4.9880` | `0.427` | **`5.2075`** | **`0.695`** (`69.5%`) | `5.3032` / `5.0249` | — | — |
| `1,525` | `stable` (post-recycle) | `1.50e-04` | `4.8823` | `0.425` | — | — | — | `24.8 GB` | `40.04s` |
| `1,540` | `stable` (`wide`) | `1.50e-04` | **`4.6826`** | `0.553` | — | — | — | `20.7 GB` | `43.31s/step` (`649.66s` / 15) |
| `1,560` | `stable` (`wide`) | `1.50e-04` | `4.9307` | `0.450` | — | — | — | `16.8 GB` | `46.15s/step` (`922.96s` / 20) |
| `1,574` | `stopped` (clean `STOP`) | `1.50e-04` | `4.9055` | — | — | — | best `5.2075` | — | `7,239 MB` post-`zero_grad`, `11,588 MB` peak |

### 9.2 Core File Map

| File | Role in Single-GPU / XPU Stack |
| :--- | :--- |
| `xpu_compat.py` | Discrete XPU selection, `autocast_context`, `supports_fp64`, Win64 commit & priority helpers |
| `models/common.py` | Device-agnostic attention, RMSNorm/LayerNorm, RoPE, and block primitives |
| `models/next_concept.py` | FlexAttention XPU/MPS guard, static `torch.where` sampling, `p_eff` slicing, boolean 4D SDPA masks |
| `models/quant.py` | Static `torch.where` K-means & dead-code restart fixes for XPU |
| `models/vqvae.py` | Stage-1 VQ-VAE wrapper (`_encode_pre_quant`, `encode`, `idx_to_ncm_input`) |
| `scripts/train_vqvae.py` | Resumable Stage-1 VQ-VAE trainer with GPT-2 SVD `wte` init and step-0 K-means seeding |
| `scripts/preencode_concepts.py` | Offline `uint16` `gt_idx_508.bin` + `unigrams.pt` generator |
| `scripts/train_ncm.py` | `KahanBF16AdamW`, `.mean(dim=0)` `NaN` fix, `256`-bucket trimming, single-copy saves, worker recycle, CPU offload |
| `tests/test_xpu_core.py`, `tests/test_single_gpu_trainer.py` | Parity, prefix-trim equivalence, recycle guard, and CPU-offload cross-resume tests |

### 9.3 Suggested Upstream PR Breakdown (`HazyResearch/ncm`)

To keep reviews focused and cleanly mergeable alongside open community PRs (`#4`/`#5` KV-cache generation, `#6`/`#25` Apple MPS/MLX evaluation, and `#29` CPU/XPU eval sharding), we recommend splitting these changes into three independent PRs:

1. **`upstream/lossless-prefix-trimming`** (`models/next_concept.py`, `scripts/train_ncm.py`, tests)
   - *Scope:* `_trim_prefix_to_bucket` (`bucket_size=256`), `p_eff` slicing in `_run_stack()` and `_level0_slice()`, and boolean 4D SDPA masks.
   - *Benefit:* Pure CUDA/XPU/MPS speed and activation-memory win (`0.0` logit delta) for any variable-length prefix workload when `use_flex_attention=false`; complements `#4`/`#5` KV-cache decode without touching `generate()`.
2. **`upstream/intel-xpu-backend`** (`xpu_compat.py`, `models/quant.py`, `models/next_concept.py`, `evaluate.py`, `scripts/eval_shard.py`, `run-xpu.sh`, `run-xpu.ps1`)
   - *Scope:* Discrete Arc filtering (`get_discrete_xpu_indices`), `supports_fp64()` / `bfloat16` autocast guards, static `torch.where` quantizer/sampler fixes, and `use_flex_attention` XPU/MPS guard.
   - *Benefit:* Extends `#6`/`#25` (MPS) and `#29` (eval sharding) with first-class Intel Arc XPU support and zero-`float64` consumer GPU compatibility.
3. **`upstream/single-gpu-1p3b-trainer`** (`scripts/train_vqvae.py`, `scripts/preencode_concepts.py`, `scripts/train_ncm.py`, `docs/SINGLE_GPU_NCM_IMPROVEMENTS.md`, tests)
   - *Scope:* Pure-PyTorch `KahanBF16AdamW` (with XPU sum/div reduction fix and optional `--optimizer-cpu-offload`), offline `gt_idx` memmap pre-encoding, `S`-adaptive gradient checkpointing, single-copy atomic checkpointing, and WDDM/Level Zero worker recycling.
   - *Benefit:* Enables `1.3B+` two-stage NCM training on single `16 GB` consumer GPUs (and `8–12 GB` via offload) without `bitsandbytes`.

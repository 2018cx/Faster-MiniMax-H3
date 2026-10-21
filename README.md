# MiniMax-H3-FineTuning

English | [中文](README_zh.md)

A minimal, working fine-tuning path for [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
the open-weight omni-modal model that jointly generates video **and** synchronized stereo audio.

MiniMax released the H3 weights "to support further development, including fine-tuning", but ships **no
trainer**. The Hugging Face Diffusers integration is inference-only. This repository fills the gap with a
supervised rectified-flow trainer built directly on the official Diffusers integration, plus latent-caching
preprocessors — and documents the nine hard-won fixes behind it: five H3-specific numeric conventions that
silently corrupt weights, and four LoRA × ZeRO-3 × long-sequence landmines (see [FIXES.md](FIXES.md); we
learned every one of them the hard way).

Proven at production scale: LoRA over 2,000 ~30-second clips (448×768, ~65k tokens per sequence, real
stereo audio in the loss) on 8×A800, with lockstep validation and multi-day-run guard rails.

<!-- BEGIN GENERATED OPTIMIZATION HISTORY -->
## Agent Optimization History

Measured on **8×H20**, BF16 FL2VA LoRA/ZeRO-3, batch one per rank and approximately 33,750 tokens per sample. Each speed arm has two optimizer updates: update 1 is warm-up, and the interval to update 2 is timed. **Round 19 is the final full-check accepted best for Rounds 1–40 at approximately 1.87×.** The final Round 40 compiled-SwiGLU candidate completed in Round 41 without beating it, triggering the user-authorized fallback selection.

![Rounds 1–40: candidate observations and cumulative full-check accepted best; Round 19 selected, 2× target not reached](docs/performance-views/02-history.svg)

**F** = fresh reference/candidate/reference full check (numerics, benefit and integrity). **T** = uninstrumented screening time; a timing PASS alone does not accept a candidate. **P** = profile-only observation, excluded from speed claims. **—** = no valid candidate timing. The T column selects the lowest valid candidate screening latency in each round. Accepted best changes only after F; 1.000000× in rounds 1–3 denotes the unchanged reference anchor.

| Round | Experiment | Best T: seconds / ratio | F or other result | Accepted best at round end |
|---:|---|---:|---|---:|
| 1 | Lock preflight | — | **Failure; no candidate time.** Lock permission error before GPU acquisition; CPU quick passed; no valid timing. | 1.000000× |
| 2 | Unchanged baseline | — | **P only**: 117.026331 s; no candidate time. Baseline observation only; no optimization or full acceptance. | 1.000000× |
| 3 | Nsight baseline observation | — | **P only**: 117.356360 s; no candidate time. Instrumented profile only; raw trace import failed (missing libdw); no optimization. | 1.000000× |
| 4 | Native cuDNN SDPA | 87.829333 / 1.333024× | **F PASS**: 88.074620 s / 1.326490×. Accepted native backend substitution; fresh full check followed the round. | 1.326490× |
| 5 | Strided cuDNN inputs | 87.195134 / 1.342719× | **T only; not accepted.** Rejected: approximately 0.72% versus prior screen, below noise resolution; profiler failure retried separately. | 1.326490× |
| 6 | Checkpoint attention O/LSE retention | 76.372563 / 1.532993× | **F PASS**: 76.148855 s / 1.534143×. Accepted checkpoint-scoped native attention reuse; all 108 tensors passed. | 1.534143× |
| 7 | Memory-efficient attention | 150.820727 / 0.776277× | **T only; not accepted.** Rejected: substantially slower; timing structure PASS is not numerical acceptance. | 1.534143× |
| 8 | Compiled RMSNorm | 73.203333 / 1.599362× | **F PASS**: 73.052049 s / 1.601428×. Accepted; first compile/cache failure retried; no measured-window recompilation. Full check logged compiler cache hits; do not assume every artifact was cold. | 1.601428× |
| 9 | FlexAttention backends | 87.035622 / 1.345180× | **T only; not accepted.** Rejected: standard Flex slower; experimental TMA failed numerics; backend-name launch error has no timing. | 1.601428× |
| 10 | Initial frozen-Linear cache | — | **Failure; no candidate time.** No candidate attempt completed update 1: a stream-lifetime fix and busy preflights preceded a rank SIGKILL. Pinned allocation pressure is suspected. | 1.601428× |
| 11 | Bounded native Linear cache | 69.011473 / 1.696509× | **F PASS**: 69.093446 s / 1.696914×. Accepted 16 GiB GPU / 64 GiB host cache; full check authoritative despite slightly faster standalone screen. | 1.696914× |
| 12 | Fused RoPE and layout | 68.582903 / 1.707110× | **F PASS**: 67.394364 s / 1.733697×. Accepted N0 rotary component / N1 stack. Full check passed the predeclared 68.848 s cutoff; screen/full-check variation is recorded separately. | 1.733697× |
| 13 | cuBLASLt/workspace search | 69.000824 / 1.696771× | **T only; not accepted.** Rejected both default and 128 MiB workspace settings; neither beat accepted cuBLAS. | 1.733697× |
| 14 | Host cache 96 GiB | 66.105282 / 1.771093× | **F PASS**: 66.096057 s / 1.770093×. Accepted 96 GiB host budget; the 160 GiB trial ended in SIGKILL before update 1, without a valid time. | 1.770093× |
| 15 | Packed frozen QKV | 70.068271 / 1.670922× | **T only; not accepted.** Rejected native cuBLAS and cuBLASLt packed variants; both slower. | 1.770093× |
| 16 | Power-of-two pinned slabs | 65.440259 / 1.789091× | **F PASS**: 64.746800 s / 1.808602×. Accepted 176 GiB host / 24 GiB GPU slabs, with all 700 replays. The 112 GiB screen was slower; full check passed the 65.75 s cutoff. | 1.808602× |
| 17 | NUMA placement | 67.278752 / 1.740202× | **T only; not accepted.** Rejected: slower; instrumented operator durations are not additive critical-path evidence. | 1.808602× |
| 18 | DeepSpeed communication overlap | 79.312700 / 1.476165× | **T only; not accepted.** Both overlap trials rejected; even the best was slower. Extra headroom alone did not recover performance. | 1.808602× |
| **19 · selected** | Cache headroom: 16 GiB GPU / 184 GiB host | 62.636536 / 1.869174× | **F PASS**: 62.621896 s / 1.869055×. Selected exact protected commit `fe2ccd6`; all 700 expected cache replays, no misses. | **1.869055×** |
| 20 | Grouped block parameter fetch | 62.586782 / 1.870660× | **T only; not accepted.** Only 0.056% lower latency than the accepted result: unresolved, with no full check or new accepted best. | 1.869055× |
| 21 | Sequential/grouped fetch | 62.670310 / 1.868167× | **T only; not accepted.** Rejected: no gain. All-reduce alternative rejected before timing for unsupported dtype/padding assumptions. | 1.869055× |
| 22 | Compiled modulated RMSNorm | 62.412514 / 1.875883× | **T only; not accepted.** Only 0.334% lower latency; missed the predeclared 61.5 s cutoff. No full check. | 1.869055× |
| 23 | Attention head folding | 64.016940 / 1.828869× | **T only; not accepted.** Head group 1 was slower; group 8 had component PASS but no timing. Nsight trace import failed. | 1.869055× |
| 24 | Repaired native Flex TMA components | — | **No candidate time** (accepted-best P only). Component/operator/opcheck PASS only; no candidate end-to-end time. Accepted-best profile is observation only. | 1.869055× |
| 25 | Flex TMA/CuDNN hybrid forward | 65.424116 / 1.789533× | **T only; not accepted.** Rejected default and tile-128 configurations; actual fast paths were active but slower. | 1.869055× |
| 26 | Small parameter persistence | 62.712242 / 1.866918× | **T only; not accepted.** Rejected: persistence fast path active but no gain; independent updated-shard freshness passed. | 1.869055× |
| 27 | Native fetch/prefetch configuration | 62.962044 / 1.859511× | **T only; not accepted.** Rejected: grouped leaf failed fixed FP64 norm bound (3.2577 > 3); corrected no-leaf trial passed numerics but was slower. | 1.869055× |
| 28 | Classic cuBLAS 128 MiB workspace | 62.736147 / 1.866206× | **T only; not accepted.** Rejected: production/training numerical gates passed; frozen timing showed no gain. | 1.869055× |
| 29 | Frozen Linear backward weight layout | 62.956861 / 1.859664× | **T only; not accepted.** Rejected: native contiguous weight-transpose gradient fast path active but slower. | 1.869055× |
| 30 | Native cuDNN plan wrapper | — | **No candidate time** (accepted-best P only). Component-only engine 1/index 1 PASS; no trainer timing. The default wrapper selected the existing engine, not a new gain. | 1.869055× |
| 31 | cuDNN engine 1/index 1 | 94.079174 / 1.244469× | **T only; not accepted.** Rejected: actual selected native backward plan active but much slower. | 1.869055× |
| 32 | cuDNN engine 2 knob inventory | — | **No candidate time** (accepted-best P only). Only existing config 26 supported among 0..35; no new eligible candidate and no candidate timing. | 1.869055× |
| 33 | cuDNN engine 0/3 probes | — | **Failure; no candidate time.** Engine 0 unsupported; engine 3/config 31 built but execution returned CUDNN_STATUS_INTERNAL_ERROR; no candidate timing. | 1.869055× |
| 34 | Native FA3 compiler/ABI feasibility | — | **No candidate time** (accepted-best P only). Compiler/ABI feasibility only, with corrected build issues; no native attention execution or candidate timing. | 1.869055× |
| 35 | Native FA3 forward runtime | — | **Failure; no candidate time.** FP64/single-thread component PASS, but concurrent cold-call wrapper failed context/factory gates; no integration or candidate timing. | 1.869055× |
| 36 | Native FA3 forward/CuDNN backward | 62.912493 / 1.860975× | **T only; not accepted.** Rejected: full production path active, no gain; no full check. | 1.869055× |
| 37 | Native FA3 backward components | — | **No candidate time** (accepted-best P only). ABI/TMA/factory/runtime/FP64 component gates passed; no trainer integration or candidate timing. | 1.869055× |
| 38 | Native FA3 M80 backward | 63.850828 / 1.833627× | **T only; not accepted.** Rejected: matched eight-rank numerical fixture passed; native production backward active but slower. | 1.869055× |
| 39 | Native FA3 M64 backward | 66.227518 / 1.767824× | **T only; not accepted.** Rejected: native M64 path active but slower; fast-exp/global-fast-math hypothesis rejected before policy change. | 1.869055× |
| 40 | Compiled packed SwiGLU | 62.639154 / 1.869096× (R41) | **T only; not accepted.** Repaired FP32 SiLU opmath passed component/integration gates. No timing in R40; the terminal R41 screen was slower and triggered fallback. No new full-check best. | 1.869055× |

R40 includes its candidate’s terminal R41 screen; no new accepted best followed. Full-check ratios use their fresh paired-reference mean; screening ratios use the frozen reference. Profile observations and failed/missing runs never advance the accepted curve. See [data and artifact hashes](docs/performance-data/rounds-01-40.json), [denominators, cutoffs and selection audit](docs/performance-audit.md), and the [generator](docs/performance-views/generate_history.py).

Reproduce the figure and table with `python docs/performance-views/generate_history.py`; verify checked-in outputs with `python docs/performance-views/generate_history.py --check`.

Separate matched 50-update, three-seed trajectory gate: **FAIL**. See the [validation method and results](docs/validation-results.md).
<!-- END GENERATED OPTIMIZATION HISTORY -->

Targeted repair later selected protected `5c47dee` at **1.8570097438×** in the Round 60 closeout. Its fresh six-arm 50-update validation is **running**; see the [repair evidence and selection](docs/repair-results.md).

## How it works

**Two-stage design** (so the 33B transformer is the only thing in GPU memory during training):

1. **`prepare_cache.py`** — encodes each training sample offline: target video → H3-VisualVAE latents
   (patchified rows, cover-fit framing — never stretched), target audio → H3-AudioVAE latents
   (`--encode-audio`; zero placeholder otherwise), caption → Qwen3-VL layer-50 hidden states. One `.pt`
   file per sample. **`prepare_cache_pairs.py`** is the long-clip variant: it concatenates two
   consecutively-shot clips into one ~30 s story-continuous sample (17n+5 re-alignment, seam index kept,
   both audio tracks joined).
2. **`train.py`** — loads only the transformer (`transformer` = FL2VA variant or `transformer_ref` = Ref2VA
   variant), rebuilds the packed video/audio/text sequence layout with the official
   `build_packed_sequence` / `build_row_timesteps` utilities, and optimizes a rectified-flow MSE loss.

**Training objective.** H3's released checkpoints are guidance-distilled rectified-flow models with two
peculiarities relative to the common SD3/Wan-style convention:

- the transformer's time input is `t = 1 − σ` (t=1 clean, t=0 pure noise), and
- the transformer predicts a **data-ward velocity** `v = x₀ − ε` (the scheduler reconstructs
  `x₀ = x_t + (1−t)·v` — note the `+`).

The loss therefore noises `x_t = (1−σ)·x₀ + σ·ε` and regresses `pred → (x₀ − ε)`, with **two different σ
per step**: video and audio use separate shifted schedules (`σ = shift·u / (1+(shift−1)·u)`, shift 12.0 for
video and 3.0 for audio) sampled at the same `u`, mirroring how the two schedulers advance in lockstep at
inference. Real cached audio enters the loss at weight 1.0; zero-placeholder audio is trained at weight 0
so the audio head is untouched but gradients stay synchronized ([FIXES.md](FIXES.md) #3).

**Validation.** Samples whose `md5(sample_id) % 20 == 0` (~5%) form a deterministic val split. Every
`--val-every` steps, all ranks run a lockstep evaluation on a fixed σ grid {0.3, 0.6, 0.9} with
per-(file, σ) seeded noise — the val curve is a fixed function of the weights, comparable across steps and
runs. `plot_loss.py` renders train + per-modality + VAL curves from `train.log`.

**Trainable-parameter modes:**

| `--trainable` | What trains | Use case |
|---|---|---|
| `heads` | `proj_out` + `audio_proj_out` (~1M params) | smoke test the whole pipeline on one GPU |
| `lora` (default) | PEFT LoRA r=16 on `to_qkv`, `to_out.0`, `linear_1`, `linear_2` (~10.7M params) | practical single-node fine-tuning |
| `all` | full 33B | requires `--strategy deepspeed` (ZeRO-3 config included) |

Checkpoints store only trainable tensors for `heads`/`lora` (a few MB instead of ~66 GB), gathered from
ZeRO-3 shards; `--resume` loads them weight-only *before* `deepspeed.initialize` (FIXES.md #5/#9).

## Quickstart

```bash
# 1. Environment: Python 3.11, torch >= 2.8. The H3 classes are NOT in a released
#    diffusers wheel yet — install_env.sh installs the pinned integration revision.
bash install_env.sh

# 2. Write a manifest (schema + placeholder samples in examples/)
python validate_manifest.py --manifest path/to/train.jsonl --check-files

# 3. Cache latents (debug scale: 256x256, 22 frames; raise for real runs)
python prepare_cache.py \
  --metadata path/to/train.jsonl \
  --output cache/train \
  --model /path/to/MiniMax-H3 \
  --height 256 --width 256 --frames 22 --encode-text --encode-audio

# 4. Train (single GPU smoke)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/smoke \
  --max-steps 10 --trainable heads --strategy ddp

# 4'. Multi-GPU LoRA (8x, DeepSpeed ZeRO-3) — the production path
deepspeed --num_gpus 8 train.py \
  --model /path/to/MiniMax-H3 --variant ref2va \
  --cache cache/train --output runs/lora \
  --max-steps 800 --trainable lora --strategy deepspeed
```

Constraints inherited from the model: `num_frames % 17 == 5`, height/width divisible by 32, 24 fps video,
32 kHz stereo audio, `audio_latent_frames = round(num_frames / 24 * 40)`.

CUDA training defaults to cuDNN SDPA. With DeepSpeed, `--attention-cache-gib 32`
retains attention outputs and softmax statistics inside activation checkpoints,
avoiding their recomputation during backward. This uses additional GPU memory;
set `--attention-cache-gib 0` to recompute all attention, or lower the budget for
longer sequences. The cache is discarded during backward and never reused across
training steps. `--attention-backend native` restores automatic SDPA selection.

CUDA RMSNorm uses PyTorch compilation with eager precision casts preserved.
Use `--no-compile-norms` for eager execution. By default, each process owns fresh
temporary compiler caches and removes them on exit; explicit `TRITON_CACHE_DIR`
and `TORCHINDUCTOR_CACHE_DIR` settings take precedence.

DeepSpeed LoRA training also retains frozen Linear outputs within each activation
checkpoint using native PyTorch selective checkpointing. The defaults are
`--linear-cache-gpu-gib 16`, `--linear-cache-cpu-gib 184` per rank, and
`--linear-cache-max-entry-gib 2`. Larger outputs or budget misses recompute
normally. Host storage uses a fixed pinned arena with asynchronous copies and
prefetch; values are released during backward and never reused across updates.
Choose budgets for the available GPU and host memory; setting both storage
budgets to zero disables this cache. Final counters on every rank report actual
cache use, recomputation, live entries, and peak GPU allocation.

The cuDNN checkpoint path fuses rotary embedding and input layout with explicit
eager rounding and a first-order autograd kernel. `--no-compile-rope` restores
the native formula. Each new kernel variant checks its first output against the
native result; a mismatch disables the fused path and is logged. Unsupported
inputs, including trainable rotary tables, use the native formula with counted
fallbacks. Higher-order gradients are not supported by the fused operator.

**Sequence-length budget (8×A800-80GB, full attention + LoRA + ZeRO-3):** the measured ceiling is ≈**70k
tokens** per sample (65k ⇒ 76 GB steady; 76k and above OOM). Size resolution × duration against it —
448×768 @ 27 s ≈ 65k fits; 576×1024 @ 30 s ≈ 127k does not. Two knobs enforce the budget without
re-encoding caches: `H3_MAX_LF=<n>` (env) truncates cached samples to *n* video latent frames at load
time, and `--max-seq-tokens` (default 132000) is a pre-flight all-reduced skip gate. Details in
[FIXES.md](FIXES.md) #8.

## Does it work?

With the conventions wrong (timestep direction + velocity sign inverted), a heads-only run shows loss
*rising* (7.2 → 9.5 over 10 steps). With the fixes in this repo, the same setup trains at loss ≈ 0.3–1.0,
stable across 1000 steps on 8×A800, and the trained heads produce coherent generations when patched back
into the inference pipeline. The LoRA path runs 30-second 65k-token sequences with real audio at
~7.5–8 min/step on 8×A800 (heads at 33k tokens: ~53 s/step). Details in [FIXES.md](FIXES.md).

## Current limitations

- Reference media are not yet encoded into the conditioning sequence (H3 reference rows); captions and
  target video/audio train, reference images/audio do not condition. This is the main open TODO for
  faithful Ref2VA fine-tuning.
- No sample shuffling between epochs; batch size is fixed at 1 sequence per step.
- Sparse attention (used in H3's final training stage) is not released; training runs full attention, so
  keep the ~70k-token budget above in mind.
- `--resume` restores weights only (trainable-only checkpoints carry no optimizer state).

## Legal

Model weights are governed by the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
(territory and usage restrictions apply; fine-tuned weights are Model Derivatives). This repository contains
code only — no model weights and no training media.

The pinned host arena uses separate power-of-two slabs of at most 32 GiB.
PyTorch rounds each pinned allocation up to a power of two, so these slabs keep
the allocated bytes within the requested arena budget. Slots and free ranges
remain separate per slab; fragmentation falls back to native recomputation.
The runtime report includes slab sizes and process-wide native pinned allocator
counters, along with GPU peak allocation including prefetched tensors. On the
8-H20 workload the 184 GiB host / 16 GiB GPU configuration caches every eligible
frozen GEMM output within its checkpoint and releases every value before updates.
This split leaves more GPU headroom for temporary allocations than the former
176 GiB host / 24 GiB GPU configuration.
The arena keeps storage for reuse across updates; it never replays values from earlier updates.

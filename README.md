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
`--linear-cache-gpu-gib 16`, `--linear-cache-cpu-gib 64` per rank, and
`--linear-cache-max-entry-gib 1`. Larger outputs or budget misses recompute
normally. Host storage uses a fixed pinned arena with asynchronous copies and
prefetch; values are released during backward and never reused across updates.
Choose budgets for the available GPU and host memory; setting both storage
budgets to zero disables this cache. Runtime counters report actual cache use,
recomputation, live entries, and peak GPU allocation.

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

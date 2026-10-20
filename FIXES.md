# The nine fixes

Part I (fixes 1–5) are H3's numeric conventions — invisible at the "code runs fine" level, and every one
of them silently corrupts the weights if you get it wrong. All evidence comes from the official Diffusers
MiniMax-H3 integration source (scheduler/packing docstrings), not guesswork. Part II (fixes 6–9) are the
distributed-training landmines we hit once training moved to LoRA + ZeRO-3 + long (30 s, ~65k-token)
sequences; each one cost a crashed or silently-stalled multi-day run before it was understood.

## Part I — numeric conventions

The first version of this trainer produced a *rising* loss (7.2 → 9.5 in 10 steps, heads mode) until
fixes 1–2 landed.

## 1. Timestep direction is inverted relative to σ

`MiniMaxH3Scheduler.step` docstring:

> The current timestep, one of `self.timesteps` (so **`timestep == 1 - sigma`**).

The transformer's time input is `1 − σ` (1 = clean, 0 = pure noise). If you noise with
`x_t = (1−t)·x₀ + t·ε` — where your `t` *is* σ — and then feed that same `t` to `build_row_timesteps`,
the conditioning is exactly inverted: an almost-clean sample is labeled "almost pure noise". A telltale
inconsistency: the conditioning rows use `0.999` (≈ clean in the `1−σ` convention) while the generated
rows got raw σ.

**Fix:** pass `1.0 − σ_video` and `1.0 − σ_audio`.

## 2. The velocity target sign is opposite to the usual flow-matching convention

Same docstring:

> The model output is a **data-ward velocity**, so the denoised estimate is `x0 = x_t + (1 - t) * v` —
> **note the `+`, the opposite of the usual flow-match convention.**

So the model predicts `v = x₀ − ε`. The habitual SD3/Wan-style target `(ε − x₀)` is the *negative* of what
this model natively outputs — training on it actively pushes the weights the wrong way (hence the rising
loss).

**Fix:** regress `pred → (clean − noise)`.

## 3. Zero-placeholder audio must not train the audio head

When the cache stores zero tensors for audio, training `audio_proj_out` against
`(0 − ε)` teaches the audio head to predict pure noise. Simply *skipping* the audio loss term breaks DDP
(`audio_proj_out` receives no gradient and DDP raises).

**Fix:** multiply the audio loss by `0.0` when the cached audio is all-zero — gradients still flow
(as zeros) so DDP stays happy, and the audio head is untouched. With real cached audio (see
`prepare_cache.py --encode-audio`) the weight is 1.0 and the audio head trains normally.

## 4. Video and audio use different shifted noise schedules

At inference the two schedulers advance in lockstep with different shifts
(`σ = shift·u / (1 + (shift−1)·u)`, video shift 12.0, audio shift 3.0), so at any step σ_video ≠ σ_audio.
Training both modalities at one shared uniform `t` never visits the (σ_v, σ_a) pairings that inference
actually traverses.

**Fix:** sample one `u ~ U(0.02, 0.98)` and map it through both shift curves.

## 5. Full-state checkpointing trips the NCCL watchdog

Saving `model.state_dict()` for a 33B bf16 model writes ~66 GB per checkpoint. In heads/LoRA modes that is
all frozen weight. Worse than the disk cost: rank 0 blocks inside `torch.save` for minutes while the other
ranks wait at the next collective — after the timeout the NCCL watchdog kills the job with SIGABRT.

**Fix:** for `heads`/`lora`, save only tensors with `requires_grad=True` (a few MB). Under ZeRO-3 the
trainable tensors are sharded, so gather them first (`deepspeed.zero.GatheredParameters`) and save from
rank 0. Keep full state only for `--trainable all` (where DeepSpeed handles sharded saving anyway).

## Part II — LoRA × ZeRO-3 × long-sequence landmines

## 6. torch activation checkpointing is incompatible with PEFT layers under ZeRO-3

With LoRA adapters injected and ZeRO-3 sharding active, torch's `checkpoint` — **both** reentrant and
non-reentrant variants — recomputes the forward against parameters that ZeRO has already re-partitioned,
so the recompute sees shape-`[0]` tensors and crashes. This is not fixable from the config side.

**Fix:** hand DeepSpeed's own implementation to the model:
`model.enable_gradient_checkpointing(deepspeed.checkpointing.checkpoint)` — it cooperates with the ZeRO
gather hooks. Two corollaries: keep `lora_dropout=0.0` so the recomputed forward is deterministic, and do
**not** enable `partition_activations` / `checkpoint_in_cpu` (both slice the 0-d time-embedding scalar and
crash the temb linear).

## 7. Never put logic between `engine.backward()` and `engine.step()`

DeepSpeed's ZeRO-3 backward leaves per-parameter accumulation state that `engine.step()` consumes; any
conditional skip, all-reduce, or early-`continue` inserted between the two (e.g. a reactive OOM handler)
desynchronizes that contract and later steps die with `KeyError: averaged_gradients`.

**Fix:** make OOM protection a **pre-flight gate**: compute the packed sequence length *before* the
forward pass, all-reduce the skip decision (MAX) so every rank skips the same sample together, and only
then enter forward/backward/step as an uninterrupted triple. Pair the gate with an anti-spin guard — if a
whole epoch executes zero steps, raise instead of silently looping forever (a miscalibrated threshold once
turned a training job into an infinite no-op that looked alive from the outside).

## 8. Long sequences: allocator fragmentation and the real capacity ceiling

Two separate effects, one symptom (OOM):

- **Fragmentation** — at 50k+ tokens the default allocator strands 14–16 GB of unusable slack. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before `import torch`** (an env var set after the
  import silently does nothing).
- **True capacity** — with full attention (H3's sparse-attention training stage is not released), LoRA,
  ZeRO-3 and deepspeed checkpointing on 8×A800-80GB, the measured ceiling is **≈70k tokens per sample**
  (65k ⇒ 76 GB steady; 76k/86k/100k/127k all OOM regardless of fragmentation fixes; slope ≥0.6 GB per 1k
  tokens). Budget resolution × duration against this number — e.g. 448×768 @ 27 s ≈ 65k tokens fits,
  576×1024 @ 30 s ≈ 127k does not.

The trainer's `H3_MAX_LF` env truncation and `--max-seq-tokens` gate exist to enforce this budget without
re-encoding the cache. When truncating cached samples by hand, remember the audio rows are
**channel-major** (`[ch0 × n, ch1 × n]`): slice each channel's block separately, and re-align frame counts
to `17n+5`.

## 9. Resume must load weights before `deepspeed.initialize`

After ZeRO-3 partitioning, `model.load_state_dict(..., strict=False)` sees shape-`[0]` parameter shards —
it either crashes or, with `strict=False`, *silently loads nothing* and "resumes" from random adapters.

**Fix:** load the checkpoint into the still-unsharded model **before** DDP wrapping /
`deepspeed.initialize`. Trainable-only checkpoints (fix 5) carry no optimizer state, so a resume is
weight-only: the optimizer restarts fresh, which is acceptable for AdamW at LoRA scale but worth knowing.

## Verification

Same data, same hyperparameters, heads mode, 8 GPUs:

| | loss @ step 1→10 | 1000-step behavior |
|---|---|---|
| before fixes | 7.25 → 9.54 (rising) | — |
| after fixes | ≈ 0.3–1.0 (stable, σ-dependent variance) | completes; checkpoints are ~8 MB |

A/B inference with the trained heads patched into the official pipeline (same seed, same prompt, same
references) produces coherent generations; the video head moves ~4–7% in relative norm after 1000 steps at
lr 1e-5 while the audio head stays put (weight-decay drift only), as intended with placeholder audio.

With Part II in place, the same trainer runs LoRA (10.7M params) over ~65k-token 30-second clips with real
stereo audio in the loss on 8×A800: ~7.5–8 min/step wall clock, 76 GB steady, validation on the fixed σ
grid every 100 steps. (For pace reference: heads-only at 33k tokens is ~53 s/step — full-model backward
through 33B plus checkpoint recompute is ~6–7× more expensive per token than the heads path. If a
long-sequence run is *mysteriously* slower than that, check for a competing GPU process first: another
tenant's compute-heavy kernels time-slice the SMs and show up only as "100% util but 2× slow steps".)

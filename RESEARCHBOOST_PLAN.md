# Faster-MiniMax-H3 ResearchBoost plan

## Frozen scope

- Upstream code: `IAmIronMan42/MiniMax-H3-FineTuning` at
  `74d535b2cae5cd56f9565059e6f2b6c9ddb39df4`.
- Model: `/diff/models/MiniMax/MiniMax-H3`, exposed read-only through local
  symlinks under `.assets/model`.
- Hardware: 8 NVIDIA H20 96 GB GPUs on the assigned server.
- Reference precision: BF16, as in the upstream trainer.
- Reference workload: FL2VA, LoRA rank 16 / alpha 32, DeepSpeed ZeRO-3,
  one sequence per rank, fixed synthetic cache, approximately 33.5k packed
  tokens per sample (448x768, 328 frames, 64 text rows).
- Target: at least 1.20x steady-state training-step speedup.  A fully
  validated result of at least 1.05x is an acceptable minimum when profiling
  shows that the remaining target is outside the stated constraints.

The numeric target is an explicit working assumption because the owner asked
for a faster implementation without specifying a threshold. FL2VA is frozen
because the supplied model directory contains a current-Diffusers conversion
of FL2VA (verified against the original tensor values); Ref2VA is present only
in the original checkpoint naming scheme and remains supported by the trainer
when a converted checkpoint is supplied.

## Invariants

The candidate must preserve the cached samples and their order, random-number
draw order, batch size, learning rate, optimizer and its update count, loss,
LoRA configuration, gradient-checkpointing semantics, and ZeRO stage.  Whole-
step FP8/INT8 or a BF16-to-lower-precision conversion is not allowed.  A
candidate may be accepted only by the frozen ResearchBoost judge, using
reference-candidate-reference timing in independent processes.

## Milestones

1. One-GPU, one-step heads sanity at 256x256 / 22 frames.
2. Eight-GPU LoRA baseline on a deterministic ~33.5k-token fixture.
3. Freeze environment fingerprints, reference timing, exact short-step
   recordings, trajectory tolerances, loader digest, and judge hash.
4. Run negative controls and the unmodified control arm.
5. Profile, optimize one hotspot per round, and accept only a full judge PASS.
6. Confirm the winner with a longer end-to-end run and publish all JSON logs.

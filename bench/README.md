# Faster-MiniMax-H3 frozen ResearchBoost judge

The only formal verdict is `check.json` written by `bench/run.py check`.
The judge validates the frozen inputs, acquires all eight GPUs, runs
reference → candidate → reference in independent processes, compares the
three-step loss trajectory and trainable checkpoint bitwise, then applies a
speed gate of `max(1.05, 1 + 2 * reference_spread)`.

```bash
./.venv/bin/python bench/run.py selftest --out runs/judge-selftest.json
./.venv/bin/python bench/run.py profile --arm reference --out runs/profile-reference
./.venv/bin/python bench/run.py profile --arm candidate --out runs/profile-candidate
./.venv/bin/python bench/run.py check --out runs/researchboost-checkpoint45
```

Step 1 is warm-up. The score is the median CUDA-event duration of steps 2–3.
Any input, source, fingerprint, workload, loss, or checkpoint mismatch is an
error/incorrect result rather than a speed result.

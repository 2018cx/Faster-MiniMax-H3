# Performance history evidence and selection audit

This page preserves the original Round 19 selection and failed validation. See [targeted repair and fresh validation results](repair-results.md) for the later authorized work.

This document supplies the detail behind the short **Agent Optimization History** paragraph and complete Round 1–40 table in [README.md](../README.md). The [machine-readable history](performance-data/rounds-01-40.json) retains exact values, observation types, original artifact paths and SHA256 hashes. Local run paths are provenance identifiers, not links to files shipped in the public repository.

## Measurement scope and evidence classes

The workload is real FL2VA LoRA training on eight H20 GPUs with ZeRO-3: batch one per rank, approximately 33,750 packed tokens, 448×768 video and 328 frames. Model weights/activations are BF16; losses and AdamW state retain reference FP32 precision. The 108 trainable tensors contain 10,655,744 elements. Each speed arm executes exactly two optimizer updates; update 1 warms up and the interval from its completion to update 2 completion is measured. This is a short-step benchmark, not a full-training speed claim.

| Evidence | What it establishes | What advances the accepted-best curve |
|---|---|---|
| **F: full check** | Fresh reference → candidate → reference; checkpoint numerics, benefit, frozen inputs, source and run fingerprints | Only a complete full PASS establishing a new best |
| **T: screening time** | Uninstrumented candidate latency, source/input integrity and workload structure | Nothing by itself, even when its JSON says PASS |
| **P: profile** | Observation of the current accepted version for attribution; synchronization/tracing may perturb the interval | Nothing; its interval/ratio is not a candidate speed claim |
| **Component PASS / failure / missing timing** | Operator, ABI, numerical or integration evidence; or an unsuccessful launch/gate | Nothing; do not assign zero latency or fabricate a speedup |

The README's T column selects the lowest valid completed candidate screening **latency**, not the largest ratio across different denominators. Every valid screen and profile is retained in the data, including slower configurations. F is shown separately even when its candidate latency differs from the screen. Baseline 1.000000× before the first accepted optimization is a reference anchor, not an invented full check.

## Exact Round 19 selection evidence

| Field | Authoritative evidence |
|---|---|
| Selected source | `fe2ccd6cc564b08a165793a66dce4f01faf2ea21` |
| Immutable rollback | `rollback/accepted-1.8691x` |
| Full check | `runs/target-2x/round-19/check/check.json`; complete, top-level PASS |
| Candidate | 62.62189586297609 s |
| Fresh reference mean | 117.0437853950134 s |
| Accepted speedup | **1.8690552846103328×** |
| Reference-pair spread | 0.0000492659392511203 (approximately 0.00493%) |
| Frozen benefit requirement | 1.02× |
| Complete trainable comparison | 108 tensors; max absolute difference 0.0000400543212890625; aggregate relative L2 0.00014503513111936508 |
| Numerical limits | Max absolute difference and relative L2 each ≤0.0005 |
| Checked candidate source digest | `8eaa4bb9b3354cfe51a7c3e4b1ad1fe0a8fbd8c1f075b3b5d4459766b7b562ad` |
| Independent screening observation | 62.63653637695825 s; 1.8691741215284579× using the frozen screening denominator |

The selected stack uses native cuDNN attention with checkpoint-scoped output/LSE retention, compiled RMSNorm, fused RoPE/layout and native frozen-Linear selective checkpoint caching. Round 19 selects 16 GiB GPU / 184 GiB pinned host cache budgets per rank, leaving temporary GPU headroom while retaining all 700 expected Linear replays across the two updates. No cache values survive an optimizer update. The exact selected source digest was independently matched to the full-check source before the long validation started.

## Why later apparent ratios did not supersede it

Screening ratios use the fixed **117.07859285798622 s** reference. Full-check ratios use that check's fresh paired-reference mean. A larger reported ratio can therefore accompany a slower candidate. Rounded ratios can also hide the ordering; compare the exact latencies and evidence types.

| Observation | Candidate seconds | Reported ratio | Decision |
|---|---:|---:|---|
| R19 authoritative F | 62.62189586297609 | 1.8690552846103328× | Accepted and selected |
| R20 grouped-fetch T | 62.58678177499678 | 1.8706600585230722× | Only 0.056% lower latency; unresolved gain, no full check |
| R22 modulated-norm T | 62.41251446102979 | 1.8758832882961285× | Only 0.334% lower latency; failed the predeclared **61.5 s** retention cutoff, no full check |
| R39 current-best P | 62.630348826001864 | 1.8693587861573495× | Profile of accepted code, not the candidate; actual R39 T was 66.22751799505204 s |
| R41 final SwiGLU T, originating R40 | 62.639154232048895 | 1.8690960038231768× | Slightly larger ratio but **slower** than R19; different denominator, no new full-check best |

The R20/R22 outcomes and cutoffs were recorded before acceptance; they were not retroactively changed to select Round 19. No later round established a new full-check PASS best. Later component passes and active-path counters do not change that fact.

## Final opportunity: Round 40 → Round 41

Round 40 developed the final compiled packed-SwiGLU candidate. An initial compiled derivative changed SiLU intermediate precision and was rejected before timing. The repaired version restored native FP32 SiLU opmath, passed fixed component bounds and a matched eight-rank fixture, and was archived unaccepted at the round boundary. It had **no candidate timing inside Round 40**.

The same final opportunity completed a frozen, uninstrumented screening run in Round 41. Its terminal `time/time.json` is complete with structure/input PASS and unchanged source. Every rank reported 52 installs, 204 compiled forwards and 100 compiled backwards, without fallback or measured-window recompilation. The 62.639154232048895 s result was slower than the protected 62.62189586297609 s and also missed the predeclared 62.1 s screening cutoff. It did not earn a full-check acceptance. Per the user's one-trial rule, this triggered fallback to the exact protected Round 19 commit. Round 42 profiling was cancelled and archived without a candidate interpretation; speed exploration stopped.

The R40 table row and x=40 screen marker explicitly carry **observed round 41**. This preserves the requested 1–40 history while showing the final opportunity's terminal evidence rather than pretending that a Round 40 timing existed.

## Separate matched trajectory gate

The selected fallback's matched 50-update validation completed with **FAIL**. See the [method, raw seed values, eight decisions and checkpoint drift](validation-results.md) and [machine-readable validation evidence](performance-data/validation-50-step.json). The separate two-update speed acceptance does not replace this gate. No 2× result is claimed. Both source trees remained unchanged.
At updates 25 and 50, record total/video/audio losses (equal-rank mean of native FP32 scalars) and the full preclip, unscaled global L2 over all 108 trainable tensors. Preserve exact paired initial state/input/RNG/precision/optimizer contracts, all update counts and finiteness checks, and six final trainable checkpoints with paired drift. For all eight metric/endpoint comparisons, require `abs(mean(candidate-reference) / reference_sample_SD(ddof=1)) <= 2.0` across exactly three seeds. Exactly zero SD passes only exactly zero mean difference; otherwise it fails as infinite. No epsilon or extra seeds. Local evidence is written to `runs/target-2x/validation/summary.json` and `REPORT.md`; the completed results are preserved in the linked validation documentation.

## Reproduction and visual verification

```bash
python docs/performance-views/generate_history.py
python docs/performance-views/generate_history.py --check
# Optional CPU render, using installed librsvg/Cairo on Linux:
python docs/performance-views/render_svg.py docs/performance-views/02-history.svg /tmp/h3-history.png
```

The checked-in generator consumes the checked-in JSON; no measurements are embedded only in plotting code and no GPU or new benchmark run is needed to reproduce the figure/table. It verifies all 40 round IDs, evidence classes, each ratio's denominator, the accepted curve and the R40/R41 fallback ordering. The SVG includes the data SHA256 and accessible title/description. Failure/missing markers live on a status rail and are never treated as zero speedup.

Visual language follows the [Faster-LLaVA history view](https://github.com/AI4AI-for-Anything/Faster-LLaVA/blob/main/docs/performance-views/02-history.svg): white background, burgundy accepted curve, open/filled observations, muted axes and compact milestone cards. This is original artwork adapted to the H3 evidence classes and complete 40-round axis. The inspected reference SVG hash is stored in the data. The generated SVG must be rendered and visually checked before publication.

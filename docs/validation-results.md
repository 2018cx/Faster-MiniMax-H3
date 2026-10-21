# Matched 50-update validation

This page preserves the original Round 19 selection and failed validation. See [targeted repair and the current pending validation](repair-results.md) for the later authorized work.

Selected Round 19 commit: `fe2ccd6cc564b08a165793a66dce4f01faf2ea21`. Trajectory gate: **FAIL**. The separate two-update speed check remains PASS at 1.8690552846103328×; the 2× speed target was not reached.

Six fresh eight-GPU arms ran exactly 50 optimizer updates at seeds 424242, 424243 and 424244 in reference/candidate, candidate/reference, reference/candidate order. Each pair used identical initial state, cache order, inputs, RNG, precision and optimizer contracts. All arms completed 50 applied updates with no skips and finite losses, norms, FP32 master weights and optimizer states. Identical external observation hooks were used in both versions. These instrumented run durations are not speed measurements.

Losses are equal-rank means of native FP32 scalars, aggregated in host FP64. The gradient metric is the native full preclip, unscaled global L2 over all 108 LoRA tensors (10,655,744 elements); it is already global and is recorded once after asserting rank agreement. At steps 25 and 50, the fixed gate is `abs(mean(candidate-reference) / reference_sample_SD(ddof=1)) <= 2.0` across exactly the three matched seeds. All eight comparisons must pass. Exactly zero reference SD passes only exactly zero loaded mean difference; otherwise the normalized difference is infinite and fails. There is no epsilon, extra seed, reroll or adjusted threshold.

| Step | Metric | Reference mean | Candidate mean | Reference sample SD | Candidate sample SD | Mean difference | Normalized difference | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 25 | total_loss | 2.7234065035978952 | 2.7501189907391868 | 0.21149600253896447 | 0.2020403780159814 | 0.026712487141291302 | 0.12630256279368679 | PASS |
| 25 | video_loss | 1.4406252801418304 | 1.4539728313684464 | 0.34168775290820735 | 0.35270880723053993 | 0.013347551226615906 | 0.039063592748089092 | PASS |
| 25 | audio_loss | 1.2827812284231186 | 1.2961461593707402 | 0.19400755855499524 | 0.18885954171482858 | 0.013364930947621664 | 0.068888712621127654 | PASS |
| 25 | gradient_l2 | 10.683403227951811 | 895829.90016399533 | 6.2666468983130228 | 1551605.0640231012 | 895819.21676076739 | 142950.32595532411 | FAIL |
| 50 | total_loss | 5.9046637813250227 | 5.6409096519152326 | 2.6334222691638538 | 2.2565328281907684 | -0.26375412940979004 | -0.10015641338581656 | PASS |
| 50 | video_loss | 3.3613701413075128 | 3.129277085264524 | 2.2861582208290647 | 1.9095124291150516 | -0.2320930560429891 | -0.10152099444754162 | PASS |
| 50 | audio_loss | 2.5432936946551004 | 2.5116325616836548 | 0.6296773373187321 | 0.63042014198417662 | -0.031661132971445717 | -0.050281518954237645 | PASS |
| 50 | gradient_l2 | 234.03558613813607 | 109.88229213566197 | 196.83712503096254 | 76.700905563738004 | -124.15329400247408 | -0.63074124854721758 | PASS |

The failed step-25 gradient comparison is dominated by seed 424242: reference L2 15.870537215785967 versus candidate 2687469.102919533. All six loss comparisons and the step-50 gradient comparison pass the fixed rule. The complete trajectories also contain intermittent large finite preclip gradient spikes. The table below shows each seed’s largest candidate norm over all 50 updates as descriptive evidence, without adding an acceptance threshold.

| Seed | Update | Reference gradient L2 | Candidate gradient L2 |
|---:|---:|---:|---:|
| 424242 | 49 | 62.427262508159203 | 472824853.20786852 |
| 424243 | 18 | 5.9513447860077999 | 11248900.41134833 |
| 424244 | 17 | 2.0293957480216669 | 22480633045423.355 |

Paired initial states, actual inputs, RNG, precision and optimizer contracts match. Both versions used native gradient clipping at 1.0 and completed finite updates. The selected stack contains several changed numerical implementations; these observations do not isolate a responsible component. No further ablation or optimization was run during the original fallback phase; later targeted repair is documented separately. See the [diagnostic evidence](performance-data/validation-diagnosis.json).


Raw endpoint values below retain enough digits to round-trip the recorded floats. The [JSON evidence](performance-data/validation-50-step.json) also retains every paired difference, checkpoint hash and per-tensor drift.

| Seed | Step | Metric | Reference | Candidate | Candidate − reference |
|---:|---:|---|---:|---:|---:|
| 424242 | 25 | total_loss | 2.9500587582588196 | 2.97628253698349 | 0.02622377872467041 |
| 424243 | 25 | total_loss | 2.6888339519500732 | 2.6866139471530914 | -0.0022200047969818115 |
| 424244 | 25 | total_loss | 2.5313268005847931 | 2.5874604880809784 | 0.056133687496185303 |
| 424242 | 25 | video_loss | 1.8301263004541397 | 1.8545220643281937 | 0.024395763874053955 |
| 424243 | 25 | video_loss | 1.1914024204015732 | 1.1898723840713501 | -0.0015300363302230835 |
| 424244 | 25 | video_loss | 1.3003471195697784 | 1.3175240457057953 | 0.017176926136016846 |
| 424242 | 25 | audio_loss | 1.1199324578046799 | 1.1217604279518127 | 0.0018279701471328735 |
| 424243 | 25 | audio_loss | 1.4974315166473389 | 1.4967415779829025 | -0.00068993866443634033 |
| 424244 | 25 | audio_loss | 1.230979710817337 | 1.2699364721775055 | 0.038956761360168457 |
| 424242 | 25 | gradient_l2 | 15.870537215785967 | 2687469.1029195329 | 2687453.2323823171 |
| 424243 | 25 | gradient_l2 | 3.720500576028126 | 2.2826322458354342 | -1.4378683301926918 |
| 424244 | 25 | gradient_l2 | 12.459171892041338 | 18.314940207409801 | 5.8557683153684632 |
| 424242 | 50 | total_loss | 3.2059049010276794 | 3.2064833641052246 | 0.00057846307754516602 |
| 424243 | 50 | total_loss | 6.0406025648117065 | 6.0536870360374451 | 0.013084471225738525 |
| 424244 | 50 | total_loss | 8.4674838781356812 | 7.6625585556030273 | -0.80492532253265381 |
| 424242 | 50 | video_loss | 1.3310770243406296 | 1.3331366926431656 | 0.0020596683025360107 |
| 424243 | 50 | video_loss | 2.9153692722320557 | 2.9198060929775238 | 0.0044368207454681396 |
| 424244 | 50 | video_loss | 5.8376641273498535 | 5.1348884701728821 | -0.70277565717697144 |
| 424242 | 50 | audio_loss | 1.8748278915882111 | 1.8733466565608978 | -0.0014812350273132324 |
| 424243 | 50 | audio_loss | 3.125233381986618 | 3.1338809132575989 | 0.008647531270980835 |
| 424244 | 50 | audio_loss | 2.6298198103904724 | 2.5276701152324677 | -0.10214969515800476 |
| 424242 | 50 | gradient_l2 | 11.29801719397828 | 22.810019931668975 | 11.512002737690695 |
| 424243 | 50 | gradient_l2 | 306.21463565206943 | 139.3883364980166 | -166.82629915405283 |
| 424244 | 50 | gradient_l2 | 384.59410556836048 | 167.44851997730035 | -217.14558559106013 |

| Seed | Final checkpoint max abs drift | L2 drift | Relative L2 drift |
|---:|---:|---:|---:|
| 424242 | 0.000732421875 | 0.27602133580921917 | 0.0017950933350458173 |
| 424243 | 0.00048828125 | 0.27115468247506286 | 0.0017631064600618287 |
| 424244 | 0.00048828125 | 0.27609827014475347 | 0.0017953937021225135 |

Checkpoint drift is reported rather than subjected to a new post-hoc threshold. The eight endpoint statistics above decide the short-trajectory gate. Independent CPU auditing reloaded all 2,400 per-rank update records, initial/completion files and six final checkpoints, verified paired contracts, and recomputed all eight decisions from raw values.

The selected fallback fails the required trajectory gate. No publication or completion is authorized by this result. The clean protected Round 19 speed rollback remains available, but its short two-update checkpoint PASS does not establish 50-update equivalence. Speed exploration stopped after the final Round 40/41 trial. During that original phase, the observed limitation was retained without changing the gate or trying another optimization round. Later targeted repair preserves the same failed evidence and unchanged gate.

Validation resource estimate: 8.1341 wall hours on eight GPUs (65.0726 GPU-hours), measured from driver launch to terminal summary. This includes startup, compilation, all arms, checkpointing and inter-arm gaps, plus brief pre-lock setup. It is resource accounting rather than a performance result.

Original summary SHA256: `c43d8e4da32bd85386419df2767609db063385207df837a5516e0984e5eb9394`.

[Return to the performance history and selection audit](performance-audit.md).

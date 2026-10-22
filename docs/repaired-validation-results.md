# Repaired six-arm validation results

The frozen validator reports **PASS** for exact repaired commit `5c47dee7974a8b69990f05ab4e32cf298bf4ee7e`, protected by `rollback/accepted-1.8570097438x`. All six fresh eight-GPU arms completed exactly 50 optimizer updates, without skipped or nonfinite updates and with paired initialization, input, RNG, precision and optimizer contracts intact. These instrumented durations are not speed measurements.

The independent raw-record consumer reproduced all eight decisions from 2,400 records. Checkpoint-content, source, original-evidence and publication audits are separate requirements. The authoritative frozen speed check remains **1.8570097438259385×** (candidate 63.176236236 s/update; bracketed reference mean 117.318886269 s/update).

The fixed rule is `abs(mean(candidate-reference) / reference sample SD(ddof=1)) <= 2.0`, with exact zero-SD handling and no epsilon. Seeds are 424242, 424243 and 424244. This is a short cached-workload trajectory check, with no broader model-quality claim.

| Step | Metric | Reference mean | Candidate mean | Reference sample SD | Mean difference | Standardized difference | Gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 25 | total_loss | 2.72676844398 | 2.72038586934 | 0.202430666101 | -0.00638257463773 | -0.0315296825361 | PASS |
| 25 | video_loss | 1.44019123415 | 1.44022076825 | 0.339737819122 | 2.95341014862e-05 | 8.69320394254e-05 | PASS |
| 25 | audio_loss | 1.28657720983 | 1.28016510109 | 0.192073119107 | -0.00641210873922 | -0.0333836862181 | PASS |
| 25 | gradient_l2 | 10.7943358938 | 10.2586944083 | 7.83362967336 | -0.535641485504 | -0.0683771773544 | PASS |
| 50 | total_loss | 5.6073388358 | 6.07448438803 | 2.43848044729 | 0.467145552238 | 0.191572400245 | PASS |
| 50 | video_loss | 3.26705343028 | 3.1382668068 | 2.08241916688 | -0.128786623478 | -0.0618447167249 | PASS |
| 50 | audio_loss | 2.34028549492 | 2.9362175862 | 0.405446087412 | 0.595932091276 | 1.46981833042 | PASS |
| 50 | gradient_l2 | 105.694602977 | 127.045084203 | 120.461781395 | 21.3504812258 | 0.177238631029 | PASS |

| Seed | Step | Metric | Reference | Candidate | Difference |
|---:|---:|---|---:|---:|---:|
| 424242 | 25 | total_loss | 2.94686830044 | 2.94816440344 | 0.00129610300064 |
| 424243 | 25 | total_loss | 2.68487432599 | 2.68545350432 | 0.000579178333282 |
| 424244 | 25 | total_loss | 2.54856270552 | 2.52753970027 | -0.0210230052471 |
| 424242 | 25 | video_loss | 1.82652777433 | 1.82615476847 | -0.000373005867004 |
| 424243 | 25 | video_loss | 1.18803276122 | 1.19305567443 | 0.00502291321754 |
| 424244 | 25 | video_loss | 1.3060131669 | 1.30145186186 | -0.00456130504608 |
| 424242 | 25 | audio_loss | 1.12034054101 | 1.12200963497 | 0.00166909396648 |
| 424243 | 25 | audio_loss | 1.49684156477 | 1.49239785969 | -0.00444370508194 |
| 424244 | 25 | audio_loss | 1.24254952371 | 1.22608780861 | -0.0164617151022 |
| 424242 | 25 | gradient_l2 | 13.4531416675 | 13.4111323392 | -0.0420093282618 |
| 424243 | 25 | gradient_l2 | 1.97735544459 | 6.30433771071 | 4.32698226612 |
| 424244 | 25 | gradient_l2 | 16.9525105693 | 11.060613175 | -5.89189739437 |
| 424242 | 50 | total_loss | 3.20546934009 | 3.68648743629 | 0.481018096209 |
| 424243 | 50 | total_loss | 5.53569585085 | 5.82768017054 | 0.291984319687 |
| 424244 | 50 | total_loss | 8.08085131645 | 8.70928555727 | 0.628434240818 |
| 424242 | 50 | video_loss | 1.33138684928 | 1.32240234315 | -0.00898450613022 |
| 424243 | 50 | video_loss | 2.99942496419 | 2.90688470006 | -0.0925402641296 |
| 424244 | 50 | video_loss | 5.47034847736 | 5.18551337719 | -0.284835100174 |
| 424242 | 50 | audio_loss | 1.8740824908 | 2.36408507824 | 0.490002587438 |
| 424243 | 50 | audio_loss | 2.53627097607 | 2.92079541087 | 0.384524434805 |
| 424244 | 50 | audio_loss | 2.6105030179 | 3.52377226949 | 0.913269251585 |
| 424242 | 50 | gradient_l2 | 41.429266241 | 57.9572104436 | 16.5279442026 |
| 424243 | 50 | gradient_l2 | 30.993214754 | 59.642481982 | 28.649267228 |
| 424244 | 50 | gradient_l2 | 244.661327935 | 263.535560182 | 18.8742322467 |

Final checkpoint drift is descriptive, without an additional acceptance threshold. All six checkpoints require independent CPU reload of their 108 FP32 trainables. Runtime assertions checked finite FP32 master/Adam states every update; those optimizer tensors are not serialized for independent reload.

| Seed | Maximum absolute drift | L2 drift | Relative L2 drift |
|---:|---:|---:|---:|
| 424242 | 0.00048828125 | 0.206090409047 | 0.00134030209524 |
| 424243 | 0.00048828125 | 0.274870490601 | 0.00178724756498 |
| 424244 | 0.00048828125 | 0.198990925212 | 0.00129398542603 |

Frozen summary SHA256: `950608b8819ea71932bbfae330c5376d9c7f288a5f604da35bee7f556ff8b813`. Frozen observer SHA256: `bcc856c9b344f3120245d58b8fba60b9f6fab8584b7eb688a44bf9a2504c2080`.

[Complete machine-readable values and per-tensor checkpoint drift](performance-data/repaired-validation.json). The original failed validation remains unchanged in [its historical report](validation-results.md).

"""Frozen ResearchBoost verdict logic for Faster-MiniMax-H3.

This module deliberately has no training-code imports.  It consumes records
produced by ``bench/run.py`` and is content-hashed in ``bench/frozen.json``.
Only its returned verdict is considered a benchmark result.
"""
from __future__ import annotations

import copy
import math
import statistics


REQUIRED_METRICS = {
    "step", "loss", "loss_v", "loss_a", "seq", "frames", "sigma_v",
    "audio_weight", "step_seconds", "peak_allocated_bytes",
}


def _correctness(reference_a, candidate, reference_b, checkpoint_equal):
    runs = (reference_a, candidate, reference_b)
    if any(len(run.get("metrics", ())) != 3 for run in runs):
        return "ERROR", "each arm must contain exactly three completed steps"
    for run in runs:
        for row in run["metrics"]:
            if set(row) != REQUIRED_METRICS:
                return "ERROR", "metric schema mismatch"
            numeric = [row[k] for k in ("loss", "loss_v", "loss_a", "sigma_v", "step_seconds")]
            if not all(math.isfinite(float(x)) for x in numeric):
                return "INCORRECT", "non-finite training metric"
            if row["seq"] != 33750 or row["frames"] != 328 or row["audio_weight"] != 1.0:
                return "INCORRECT", "frozen workload semantics changed"
    invariant_keys = ("step", "loss", "loss_v", "loss_a", "seq", "frames", "sigma_v", "audio_weight")
    reference_rows = reference_a["metrics"]
    for other in (candidate["metrics"], reference_b["metrics"]):
        for expected, actual in zip(reference_rows, other):
            if any(expected[k] != actual[k] for k in invariant_keys):
                return "INCORRECT", "loss trajectory or sampled workload is not bitwise-equivalent"
    if checkpoint_equal is None:
        return "ERROR", "checkpoint comparison missing"
    if not checkpoint_equal:
        return "INCORRECT", "trainable checkpoint tensors exceed the frozen N1 gate"
    return "PASS", "bitwise-equivalent loss trajectory; trainable checkpoint passes the frozen N1 gate"


def evaluate(reference_a, candidate, reference_b, checkpoint_equal, fingerprints_equal=True,
             inputs_equal=True, min_speedup=1.05):
    """Return one terminal verdict: PASS, NO_BENEFIT, INCORRECT, or ERROR."""
    if not fingerprints_equal:
        return {"verdict": "ERROR", "reason": "run fingerprints differ"}
    if not inputs_equal:
        return {"verdict": "ERROR", "reason": "frozen input fingerprint mismatch"}
    correctness, reason = _correctness(reference_a, candidate, reference_b, checkpoint_equal)
    if correctness != "PASS":
        return {"verdict": correctness, "reason": reason}

    # Step 1 includes allocator/kernel warmup.  The frozen score is the median
    # of steps 2-3, using the two bracketing references to estimate drift.
    def steady(run):
        return statistics.median(float(x["step_seconds"]) for x in run["metrics"][1:])

    ref_a_s, cand_s, ref_b_s = steady(reference_a), steady(candidate), steady(reference_b)
    ref_s = statistics.median((ref_a_s, ref_b_s))
    spread = abs(ref_a_s - ref_b_s) / ref_s
    threshold = max(float(min_speedup), 1.0 + 2.0 * spread)
    speedup = ref_s / cand_s
    detail = {
        "reference_a_seconds": ref_a_s,
        "candidate_seconds": cand_s,
        "reference_b_seconds": ref_b_s,
        "reference_seconds": ref_s,
        "reference_spread": spread,
        "speedup": speedup,
        "required_speedup": threshold,
        "candidate_peak_gib": max(x["peak_allocated_bytes"] for x in candidate["metrics"]) / 2**30,
    }
    if speedup < threshold:
        return {"verdict": "NO_BENEFIT", "reason": "speedup does not clear the frozen noise/benefit gate", **detail}
    return {"verdict": "PASS", "reason": reason, **detail}


def selftest():
    """Negative controls: every mutation must reach the expected failure path."""
    row = {
        "step": 1, "loss": 2.0, "loss_v": 1.0, "loss_a": 1.0,
        "seq": 33750, "frames": 328, "sigma_v": 0.9, "audio_weight": 1.0,
        "step_seconds": 10.0, "peak_allocated_bytes": 10 * 2**30,
    }
    base = {"metrics": [{**row, "step": i, "step_seconds": 10.0} for i in (1, 2, 3)]}
    fast = {"metrics": [{**r, "step_seconds": 8.0} for r in base["metrics"]]}
    tests = []

    def check(name, expected, a=base, b=fast, c=base, ckpt=True, fp=True, inputs=True):
        got = evaluate(a, b, c, ckpt, fp, inputs)["verdict"]
        tests.append({"mutation": name, "expected": expected, "got": got, "ok": got == expected})

    check("legitimate_candidate", "PASS")
    check("control_arm", "NO_BENEFIT", b=base)
    bad_loss = copy.deepcopy(fast); bad_loss["metrics"][1]["loss"] += 0.01
    check("loss_trajectory", "INCORRECT", b=bad_loss)
    bad_shape = copy.deepcopy(fast); bad_shape["metrics"][0]["seq"] += 1
    check("workload_shape", "INCORRECT", b=bad_shape)
    bad_nan = copy.deepcopy(fast); bad_nan["metrics"][0]["loss"] = float("nan")
    check("nan_loss", "INCORRECT", b=bad_nan)
    missing = copy.deepcopy(fast); missing["metrics"][0].pop("loss_a")
    check("missing_metric", "ERROR", b=missing)
    check("checkpoint_tensor", "INCORRECT", ckpt=False)
    check("run_fingerprint", "ERROR", fp=False)
    check("input_fingerprint", "ERROR", inputs=False)
    return {"schema": "faster-h3-judge-selftest@1", "ok": all(x["ok"] for x in tests), "results": tests}

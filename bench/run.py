#!/usr/bin/env python3
"""Frozen ResearchBoost runner for the 8xH20 MiniMax-H3 workload."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from judge import evaluate, selftest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
FORMULATION = ROOT / "task" / "formulation.json"
FROZEN = BENCH / "frozen.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def verify_frozen_judge():
    frozen = json.loads(FROZEN.read_text())
    mismatches = []
    for rel, expected in frozen["files"].items():
        actual = sha256(ROOT / rel)
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    return {"ok": not mismatches, "mismatches": mismatches, "manifest": frozen}


def resolve_input(name, model_target):
    if name.startswith("$MODEL/"):
        return Path(model_target) / name.removeprefix("$MODEL/")
    return ROOT / name


def verify_inputs():
    spec = json.loads(FORMULATION.read_text())
    mismatches = []
    for name, expected in spec["inputs"].items():
        path = resolve_input(name, spec["model_target"])
        if not path.is_file():
            mismatches.append({"path": name, "reason": "missing"})
            continue
        st = path.stat()
        actual = sha256(path) if expected["kind"] == "sha256" else f"{st.st_size}:{st.st_mtime_ns}"
        if actual != expected["value"]:
            mismatches.append({"path": name, "expected": expected["value"], "actual": actual})
    return {"ok": not mismatches, "mismatches": mismatches}


def source_digest():
    ignored = {".git", ".venv", ".assets", "runs", "__pycache__"}
    h = hashlib.sha256()
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def gpu_fingerprint():
    query = "index,uuid,name,driver_version,memory.total"
    text = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], text=True)
    versions = {}
    for name in ("torch", "diffusers", "transformers", "deepspeed", "peft"):
        mod = __import__(name)
        versions[name] = getattr(mod, "__version__", "unknown")
    return {"gpus": [x.strip() for x in text.splitlines()], "versions": versions,
            "python": sys.version.split()[0], "world_size": 8}


def gpu_busy():
    cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    text = subprocess.check_output(cmd, text=True).strip()
    return [line for line in text.splitlines() if line.strip() and int(line.rsplit(",", 1)[1].strip()) > 1024]


@contextlib.contextmanager
def gpu_lock(path):
    with Path(path).open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        busy = gpu_busy()
        if busy:
            raise RuntimeError(f"GPUs are not idle: {busy}")
        yield


def load_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_arm(label, config, output, port, timeout):
    run_dir = output / label
    run_dir.mkdir(parents=True)
    command = [
        "bash", str(ROOT / "scripts" / "run_baseline_lora.sh"),
        "--checkpoint-blocks", str(config["checkpoint_blocks"]),
        "--max-steps", "3",
    ]
    env = os.environ.copy()
    env.update({"H3_BASELINE_OUT": str(run_dir), "MASTER_PORT": str(port)})
    started = time.monotonic()
    with (run_dir / "launcher.log").open("w") as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False)
    wall = time.monotonic() - started
    if proc.returncode:
        raise RuntimeError(f"{label} exited {proc.returncode}; see {run_dir / 'launcher.log'}")
    checkpoint = run_dir / "checkpoint-3.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{label} produced no checkpoint-3.pt")
    return {"label": label, "config": config, "command": command, "wall_seconds": wall,
            "metrics": load_metrics(run_dir), "checkpoint": str(checkpoint.relative_to(output)),
            "checkpoint_sha256": sha256(checkpoint)}


def checkpoint_equal(paths):
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    keysets = [set(s["model"]) for s in states]
    if any(keys != keysets[0] for keys in keysets[1:]):
        return False, {"reason": "key mismatch", "tensor_count": [len(k) for k in keysets]}
    gate = json.loads((ROOT / "task" / "numerical_gate.json").read_text())["gates"]
    checked = 0; max_abs = 0.0; max_relative_l2 = 0.0
    for key in sorted(keysets[0]):
        reference_a, candidate, reference_b = [s["model"][key] for s in states]
        if not torch.equal(reference_a, reference_b):
            return False, {"reason": "bracketing references are not bitwise equal", "first_mismatch": key,
                           "checked": checked}
        delta = candidate.double() - reference_a.double()
        tensor_max = float(delta.abs().max())
        reference_norm = float(reference_a.double().norm())
        relative_l2 = float(delta.norm()) / reference_norm if reference_norm else (0.0 if not tensor_max else float("inf"))
        max_abs = max(max_abs, tensor_max); max_relative_l2 = max(max_relative_l2, relative_l2)
        if tensor_max > gate["candidate_max_abs"] or relative_l2 > gate["candidate_max_relative_l2"]:
            return False, {"reason": "candidate exceeds N1 gate", "first_mismatch": key, "checked": checked,
                           "max_abs": tensor_max, "relative_l2": relative_l2, "gate": gate}
        checked += 1
    return True, {"reason": "references bitwise equal; candidate within N1 gate", "checked": checked,
                  "max_abs": max_abs, "max_relative_l2": max_relative_l2, "gate": gate}


def cmd_selftest(args):
    frozen = verify_frozen_judge()
    result = selftest()
    result["frozen_judge"] = frozen
    result["ok"] = result["ok"] and frozen["ok"]
    json_dump(args.out, result)
    print(json.dumps({"ok": result["ok"], "out": str(args.out)}))
    return 0 if result["ok"] else 3


def cmd_check(args):
    cfg = json.loads((BENCH / "bench.json").read_text())
    candidate_cfg = json.loads((ROOT / cfg["candidate_config"]).read_text())
    reference_cfg = json.loads((ROOT / cfg["reference_config"]).read_text())
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    report = {"schema": "faster-h3-check@1", "complete": False,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "arms": []}
    json_dump(output / "check.json", report)
    source_before = source_digest()
    inputs_before = verify_inputs()
    frozen = verify_frozen_judge()
    negative_controls = selftest()
    fingerprint = gpu_fingerprint()
    try:
        if not inputs_before["ok"] or not frozen["ok"] or not negative_controls["ok"]:
            raise RuntimeError("preflight gate failed")
        with gpu_lock(cfg["gpu_lock"]):
            for index, (label, arm_cfg) in enumerate((
                ("reference_a", reference_cfg), ("candidate", candidate_cfg), ("reference_b", reference_cfg)
            )):
                arm = run_arm(label, arm_cfg, output, 29620 + index, cfg["timeout_seconds_per_arm"])
                arm["fingerprint"] = fingerprint
                report["arms"].append(arm)
                json_dump(output / "check.json", report)
        paths = [output / arm["checkpoint"] for arm in report["arms"]]
        equal, equality = checkpoint_equal(paths)
        inputs_after = verify_inputs()
        source_after = source_digest()
        same_fingerprints = all(arm["fingerprint"] == fingerprint for arm in report["arms"])
        verdict = evaluate(*report["arms"], checkpoint_equal=equal,
                           fingerprints_equal=same_fingerprints and source_before == source_after,
                           inputs_equal=inputs_before["ok"] and inputs_after["ok"],
                           min_speedup=cfg["min_speedup"])
        report.update({"complete": True, "verdict": verdict, "checkpoint_comparison": equality,
                       "inputs_before": inputs_before, "inputs_after": inputs_after,
                       "source_before": source_before, "source_after": source_after,
                       "frozen_judge": frozen, "negative_controls": negative_controls,
                       "cost": {"gpu_hours": sum(a["wall_seconds"] for a in report["arms"]) * 8 / 3600}})
    except Exception as error:
        report.update({"complete": True, "verdict": {"verdict": "ERROR", "reason": f"{type(error).__name__}: {error}"},
                       "inputs_before": inputs_before, "frozen_judge": frozen,
                       "negative_controls": negative_controls})
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    json_dump(output / "check.json", report)
    print(json.dumps(report["verdict"], sort_keys=True))
    return {"PASS": 0, "NO_BENEFIT": 2, "INCORRECT": 3}.get(report["verdict"]["verdict"], 4)


def cmd_profile(args):
    cfg = json.loads((BENCH / "bench.json").read_text())
    config_path = cfg["reference_config"] if args.arm == "reference" else cfg["candidate_config"]
    arm_cfg = json.loads((ROOT / config_path).read_text())
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    preflight = {"inputs": verify_inputs(), "frozen_judge": verify_frozen_judge()}
    if not preflight["inputs"]["ok"] or not preflight["frozen_judge"]["ok"]:
        json_dump(output / "profile.json", {"verdict": "ERROR", **preflight})
        return 4
    command = ["bash", str(ROOT / "scripts" / "run_baseline_lora.sh"),
               "--checkpoint-blocks", str(arm_cfg["checkpoint_blocks"]), "--max-steps", "1",
               "--profile-block-calls"]
    env = os.environ.copy(); env.update({"H3_BASELINE_OUT": str(output / "run"), "MASTER_PORT": "29630"})
    with gpu_lock(cfg["gpu_lock"]), (output / "launcher.log").open("w") as log:
        started = time.monotonic()
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=cfg["timeout_seconds_per_arm"], check=False)
        wall = time.monotonic() - started
    if proc.returncode:
        result = {"verdict": "ERROR", "reason": f"profile exited {proc.returncode}", **preflight}
    else:
        block_profile = json.loads((output / "run" / "block_profile.json").read_text())
        result = {"verdict": "PASS", "arm": args.arm, "config": arm_cfg, "command": command,
                  "wall_seconds": wall, "profile": block_profile, **preflight}
    json_dump(output / "profile.json", result)
    print(json.dumps({"verdict": result["verdict"], "out": str(output)}))
    return 0 if result["verdict"] == "PASS" else 4


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("selftest"); test.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check"); check.add_argument("--out", type=Path, required=True)
    profile = sub.add_parser("profile"); profile.add_argument("--arm", choices=("reference", "candidate"), required=True); profile.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "selftest":
        return cmd_selftest(args)
    if args.command == "profile":
        return cmd_profile(args)
    return cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot train / per-modality / validation loss curves emitted by train.py."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

RE_STEP = re.compile(r"(?<![_A-Za-z])step=(\d+)\s+loss=([0-9.eE+-]+)(?:\s+loss_v=([0-9.eE+-]+)\s+loss_a=([0-9.eE+-]+))?")
RE_VAL = re.compile(r"VAL step=(\d+)\s+loss_v=([0-9.eE+-]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    train, val = [], []
    for line in args.log.read_text().splitlines():
        if "VAL step=" in line:
            m = RE_VAL.search(line)
            if m:
                val.append((int(m.group(1)), float(m.group(2))))
            continue
        m = RE_STEP.search(line)
        if m:
            train.append((int(m.group(1)), float(m.group(2)),
                          float(m.group(3)) if m.group(3) else None,
                          float(m.group(4)) if m.group(4) else None))
    if not train:
        raise RuntimeError(f"No loss records found in {args.log}")
    steps = [t[0] for t in train]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, [t[1] for t in train], linewidth=1.0, alpha=0.45, label="train loss")
    if any(t[2] is not None for t in train):
        plt.plot(steps, [t[2] for t in train], linewidth=1.0, alpha=0.45, label="loss_v")
        plt.plot(steps, [t[3] for t in train], linewidth=1.0, alpha=0.45, label="loss_a")
    if val:
        plt.plot([v[0] for v in val], [v[1] for v in val], marker="o", linewidth=2.0, label="VAL loss_v (fixed σ grid)")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("MiniMax-H3 training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"saved {args.output} ({len(train)} train points, {len(val)} val points)")


if __name__ == "__main__":
    main()

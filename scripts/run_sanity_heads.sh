#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${H3_MODEL:-$ROOT/.assets/model}"
CACHE="${H3_SANITY_CACHE:-$ROOT/.assets/cache/sanity}"
OUT="${H3_SANITY_OUT:-$ROOT/runs/sanity-heads}"

if [[ ! -e "$CACHE" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/tools/make_synthetic_cache.py" \
    --output "$CACHE" --world-size 1 --train-files-per-rank 1 \
    --height 256 --width 256 --num-frames 22 --text-tokens 16
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$ROOT/.venv/bin/python" "$ROOT/train.py" \
  --model "$MODEL" --variant ref2va \
  --cache "$CACHE" --output "$OUT" \
  --max-steps 1 --trainable heads --strategy ddp \
  --val-every 1000000 --ckpt-every 1000000 --max-seq-tokens 4096

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${H3_MODEL:-$ROOT/.assets/model}"
CACHE="${H3_BASELINE_CACHE:-$ROOT/.assets/cache/representative}"
OUT="${H3_BASELINE_OUT:-$ROOT/runs/baseline-lora-33k}"
PORT="${MASTER_PORT:-29583}"

if ! compgen -G "$CACHE/*.pt" >/dev/null; then
  "$ROOT/.venv/bin/python" "$ROOT/tools/make_synthetic_cache.py" \
    --output "$CACHE" --world-size 8 --train-files-per-rank 1 \
    --height 448 --width 768 --num-frames 328 --text-tokens 64
fi

if [[ -e "$OUT/checkpoint-1.pt" ]]; then
  echo "Refusing to overwrite completed baseline: $OUT/checkpoint-1.pt" >&2
  exit 2
fi
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "Refusing to use occupied master port $PORT" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "$ROOT/.venv/bin/deepspeed" --master_port "$PORT" --num_gpus 8 "$ROOT/train.py" \
  --model "$MODEL" --variant fl2va \
  --cache "$CACHE" --output "$OUT" \
  --max-steps 1 --seed 42 --trainable lora --strategy deepspeed \
  --val-every 1000000 --ckpt-every 1000000 --max-seq-tokens 40000

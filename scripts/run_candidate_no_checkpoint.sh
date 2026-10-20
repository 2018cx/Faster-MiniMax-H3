#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${H3_CANDIDATE_OUT:-$ROOT/runs/candidate-no-checkpoint-33k}"
if [[ -e "$OUT/checkpoint-1.pt" ]]; then
  echo "Refusing to overwrite completed candidate: $OUT/checkpoint-1.pt" >&2
  exit 2
fi

H3_BASELINE_OUT="$OUT" MASTER_PORT="${MASTER_PORT:-29584}" \
  bash "$ROOT/scripts/run_baseline_lora.sh" --gradient-checkpointing off

#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODE=${1:-cross-validation}

cd "$PROJECT_ROOT"
"$PYTHON_BIN" src/cellpose_pipeline.py "$MODE" \
  --annotations data/annotations \
  --config configs/cellpose_sam.json \
  --output "results/cellpose_${MODE}.json"

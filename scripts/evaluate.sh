#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$PROJECT_ROOT"
"$PYTHON_BIN" src/evaluate.py \
  --annotations data/annotations \
  --predictions results/roi_predictions \
  --output-dir results
"$PYTHON_BIN" -m unittest discover -s tests -v

#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/full_image.tif" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FULL_IMAGE=$1

"$PROJECT_ROOT/scripts/infer_cellpose_full.sh" "$FULL_IMAGE"
"$PROJECT_ROOT/scripts/infer_nnunet_full.sh" "$FULL_IMAGE"

"${PYTHON_BIN:-python3}" "$PROJECT_ROOT/src/compare_full_masks.py" \
  "$FULL_IMAGE" \
  "$PROJECT_ROOT/results/full_image/cellpose_sam_mask.tif" \
  "$PROJECT_ROOT/results/full_image/nnunet_mask.tif" \
  "$PROJECT_ROOT/results/figures/full_image_method_comparison.png"

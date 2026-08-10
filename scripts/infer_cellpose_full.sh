#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/full_image.tif" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${CELLPOSE_MODEL:-$PROJECT_ROOT/models/cpsam_v2_finetuned}
FULL_IMAGE=$1

mkdir -p "$PROJECT_ROOT/results/full_image" "$PROJECT_ROOT/results/figures"
"$PYTHON_BIN" "$PROJECT_ROOT/src/infer_cellpose_full.py" \
  "$FULL_IMAGE" "$MODEL_PATH" \
  "$PROJECT_ROOT/results/full_image/cellpose_sam_mask.tif" \
  --config "$PROJECT_ROOT/configs/cellpose_sam.json" \
  --device "${CELLPOSE_DEVICE:-auto}" \
  --scale "${CELLPOSE_SCALE:-1.0}" \
  --tile-size "${CELLPOSE_TILE_SIZE:-1536}" \
  --halo "${CELLPOSE_HALO:-64}" \
  --batch-size "${CELLPOSE_BATCH_SIZE:-32}" \
  --work-mask "$PROJECT_ROOT/work/cellpose_full_mask_work.tif" \
  --progress "$PROJECT_ROOT/work/cellpose_full_mask_progress.json"

"$PYTHON_BIN" "$PROJECT_ROOT/src/visualize_full_mask.py" \
  "$FULL_IMAGE" \
  "$PROJECT_ROOT/results/full_image/cellpose_sam_mask.tif" \
  "$PROJECT_ROOT/results/figures/cellpose_sam_full_overview.png" \
  --method "Cellpose-SAM fine-tuned"

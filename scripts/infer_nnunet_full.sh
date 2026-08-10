#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/full_image.tif" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAINER=${NNUNET_TRAINER:-nnUNetTrainer_5epochs}
DEVICE=${NNUNET_DEVICE:-mps}
FULL_IMAGE=$1

export nnUNet_raw="$PROJECT_ROOT/work/nnunet/raw"
export nnUNet_preprocessed="$PROJECT_ROOT/work/nnunet/preprocessed"
export nnUNet_results="$PROJECT_ROOT/work/nnunet/results"

INPUT_DIR="$PROJECT_ROOT/work/nnunet_full_image_input"
RAW_OUTPUT_DIR="$PROJECT_ROOT/work/nnunet_full_image_output"
mkdir -p "$INPUT_DIR" "$RAW_OUTPUT_DIR" "$PROJECT_ROOT/results/full_image"
"$PYTHON_BIN" "$PROJECT_ROOT/src/prepare_full_image.py" \
  "$FULL_IMAGE" "$INPUT_DIR/full_image_0000.tif" --scale 2.0445

nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$RAW_OUTPUT_DIR" \
  -d 502 -c 2d -f all \
  -tr "$TRAINER" -chk checkpoint_best.pth \
  -device "$DEVICE" -npp 1 -nps 1 \
  --disable_tta --not_on_device

"$PYTHON_BIN" "$PROJECT_ROOT/src/postprocess_full_mask.py" \
  "$RAW_OUTPUT_DIR/full_image.tif" \
  "$PROJECT_ROOT/results/full_image/nnunet_mask.tif" \
  --reference "$FULL_IMAGE" \
  --method "nnU-Net v2 ${TRAINER} fold-all" \
  --scale 2.0445

"$PYTHON_BIN" "$PROJECT_ROOT/src/visualize_full_mask.py" \
  "$FULL_IMAGE" \
  "$PROJECT_ROOT/results/full_image/nnunet_mask.tif" \
  "$PROJECT_ROOT/results/figures/nnunet_full_overview.png" \
  --method "nnU-Net v2"

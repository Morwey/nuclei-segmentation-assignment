#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAINER=${NNUNET_TRAINER:-nnUNetTrainer_5epochs}
DEVICE=${NNUNET_DEVICE:-mps}

export nnUNet_raw="$PROJECT_ROOT/work/nnunet/raw"
export nnUNet_preprocessed="$PROJECT_ROOT/work/nnunet/preprocessed"
export nnUNet_results="$PROJECT_ROOT/work/nnunet/results"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" src/prepare_nnunet.py --annotations data/annotations --raw-root "$nnUNet_raw"
nnUNetv2_plan_and_preprocess -d 502 -c 2d -np 1 -npfp 1 --verify_dataset_integrity
cp configs/nnunet_splits.json "$nnUNet_preprocessed/Dataset502_NucleiBinaryROI/splits_final.json"

for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 502 2d "$FOLD" -tr "$TRAINER" -device "$DEVICE" --npz
done

# The all-data checkpoint is used only to produce the unlabelled full-image mask.
nnUNetv2_train 502 2d all -tr "$TRAINER" -device "$DEVICE"

"$PYTHON_BIN" src/collect_nnunet_predictions.py \
  --annotations data/annotations \
  --result-dir "$nnUNet_results/Dataset502_NucleiBinaryROI/${TRAINER}__nnUNetPlans__2d"

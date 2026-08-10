"""Collect the five nnU-Net validation masks under the common ROI naming scheme."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import load_rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/roi_predictions/nnunet_loo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rois = load_rois(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fold in range(5):
        files = sorted((args.result_dir / f"fold_{fold}" / "validation").glob("nuclei_*.tif"))
        if len(files) != 1:
            raise RuntimeError(f"Fold {fold} should contain one validation TIFF, found {len(files)}")
        coordinate = files[0].stem.removeprefix("nuclei_")
        roi = next(
            item
            for item in rois
            if item.name.split("Y00039K4-", 1)[-1].replace("_256_256", "") == coordinate
        )
        shutil.copy2(files[0], args.output_dir / f"{roi.name}-dapi_mask.tif")
        copied += 1
    print(f"Collected {copied} masks in {args.output_dir}")


if __name__ == "__main__":
    main()

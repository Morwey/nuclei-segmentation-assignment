"""Convert the five supplied ROI pairs to an nnU-Net v2 2-D dataset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import tifffile

from common import load_rois


DATASET_NAME = "Dataset502_NucleiBinaryROI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("nnUNet_raw", "work/nnUNet_raw")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.raw_root / DATASET_NAME
    if dataset.exists():
        shutil.rmtree(dataset)
    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    rois = load_rois(args.annotations)
    for roi in rois:
        coordinate = roi.name.split("Y00039K4-", 1)[-1].replace("_256_256", "")
        case = f"nuclei_{coordinate}"
        tifffile.imwrite(images_tr / f"{case}_0000.tif", roi.image)
        tifffile.imwrite(labels_tr / f"{case}.tif", roi.mask.astype("uint8"))

    dataset_json = {
        "channel_names": {"0": "DAPI"},
        "labels": {"background": 0, "nucleus": 1},
        "numTraining": len(rois),
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(dataset)


if __name__ == "__main__":
    main()

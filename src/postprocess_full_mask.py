"""Compress and validate the full-resolution binary TIFF returned by nnU-Net."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

from common import save_binary_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional source image whose shape defines the final mask resolution.",
    )
    parser.add_argument("--method", default="nnU-Net v2 fold-all")
    parser.add_argument("--scale", type=float, default=2.0445)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask = np.asarray(tifffile.imread(args.input)) > 0
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, found {mask.shape}")
    inference_shape = list(mask.shape)
    if args.reference is not None:
        with tifffile.TiffFile(args.reference) as tif:
            target_shape = tuple(tif.series[0].shape)
        if len(target_shape) != 2:
            raise ValueError(f"Expected a 2-D reference image, found {target_shape}")
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    save_binary_mask(args.output, mask)
    check = np.asarray(tifffile.imread(args.output))
    if check.shape != mask.shape or not set(np.unique(check)).issubset({0, 255}):
        raise RuntimeError("Saved mask failed shape/value validation")
    metadata = {
        "file": args.output.name,
        "method": args.method,
        "probability_threshold": 0.5,
        "inference_scale": args.scale,
        "inference_shape_yx": inference_shape,
        "shape_yx": list(check.shape),
        "dtype": str(check.dtype),
        "values": [int(value) for value in np.unique(check)],
        "foreground_fraction": float(np.mean(check > 0)),
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        "evaluation": "Five-fold ROI cross-validation and full-image visual inspection",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()

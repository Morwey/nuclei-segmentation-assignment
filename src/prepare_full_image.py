"""Upsample the full image to the effective resolution of the annotated ROIs."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=2.0445)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = np.asarray(tifffile.imread(args.input))
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Expected a 2-D uint8 image, found {image.shape} {image.dtype}")
    height, width = image.shape
    resized = cv2.resize(
        image,
        (round(width * args.scale), round(height * args.scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        args.output,
        resized,
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "YX", "source_shape_yx": [height, width], "scale": args.scale},
    )
    print(f"{args.input.name}: {image.shape} -> {resized.shape}")


if __name__ == "__main__":
    main()

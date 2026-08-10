"""Create a compact whole-image overview and four representative crops."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


CROPS = ((9600, 384), (1152, 2688), (10368, 11136), (5376, 11904))


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99.5))
    return np.clip((image.astype(np.float32) - low) / max(float(high - low), 1.0), 0, 1)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = normalize(image)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    alpha = 0.38
    color = np.array([0.0, 0.72, 0.62], dtype=np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--method", default="Segmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = tifffile.memmap(args.image)
    mask = np.asarray(tifffile.imread(args.mask)) > 0
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} versus {mask.shape}")

    stride = max(1, int(np.ceil(max(image.shape) / 1800)))
    thumbnail_image = np.asarray(image[::stride, ::stride])
    thumbnail_mask = np.asarray(mask[::stride, ::stride])

    figure = plt.figure(figsize=(11.2, 7.4))
    grid = figure.add_gridspec(2, 4, height_ratios=(1.55, 1), hspace=0.16, wspace=0.05)
    ax = figure.add_subplot(grid[0, :])
    ax.imshow(overlay(thumbnail_image, thumbnail_mask))
    ax.set_title(
        f"{args.method} | {image.shape[1]} x {image.shape[0]} px | foreground {100 * mask.mean():.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")

    size = args.crop_size
    for index, (x, y) in enumerate(CROPS):
        crop_image = np.asarray(image[y : y + size, x : x + size])
        crop_mask = np.asarray(mask[y : y + size, x : x + size])
        crop_ax = figure.add_subplot(grid[1, index])
        crop_ax.imshow(overlay(crop_image, crop_mask))
        crop_ax.set_title(f"x={x}, y={y}", fontsize=8)
        crop_ax.axis("off")

    figure.suptitle("DAPI nucleus segmentation: full-image output", y=0.995, fontsize=12, fontweight="bold")
    figure.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.015)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()

"""Compare full-image masks at fixed locations."""
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


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    gray = normalize(image)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    foreground = np.asarray(mask, dtype=bool)
    rgb[foreground] = 0.58 * rgb[foreground] + 0.42 * np.asarray(color)
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("cellpose_mask", type=Path)
    parser.add_argument("nnunet_mask", type=Path)
    parser.add_argument("entropy_mask", type=Path, nargs="?")
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = tifffile.memmap(args.image)
    cellpose = np.asarray(tifffile.imread(args.cellpose_mask)) > 0
    nnunet = np.asarray(tifffile.imread(args.nnunet_mask)) > 0
    entropy = (
        np.asarray(tifffile.imread(args.entropy_mask)) > 0
        if args.entropy_mask is not None
        else None
    )
    if (
        image.shape != cellpose.shape
        or image.shape != nnunet.shape
        or (entropy is not None and image.shape != entropy.shape)
    ):
        raise ValueError(
            f"Shape mismatch: image={image.shape}, Cellpose-SAM={cellpose.shape}, "
            f"nnU-Net={nnunet.shape}, entropy={None if entropy is None else entropy.shape}"
        )

    columns = 4 if entropy is not None else 3
    figure, axes = plt.subplots(4, columns, figsize=(13.6 if entropy is not None else 10.5, 13.0))
    size = args.crop_size
    for row, (x, y) in enumerate(CROPS):
        crop_image = np.asarray(image[y : y + size, x : x + size])
        crop_cellpose = cellpose[y : y + size, x : x + size]
        crop_nnunet = nnunet[y : y + size, x : x + size]
        axes[row, 0].imshow(normalize(crop_image), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(overlay(crop_image, crop_cellpose, (0.0, 0.72, 0.62)))
        axes[row, 2].imshow(overlay(crop_image, crop_nnunet, (0.95, 0.53, 0.15)))
        if entropy is not None:
            crop_entropy = entropy[y : y + size, x : x + size]
            axes[row, 3].imshow(overlay(crop_image, crop_entropy, (0.55, 0.35, 0.85)))
        axes[row, 0].set_ylabel(f"x={x}\ny={y}", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("DAPI", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Cellpose-SAM", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("nnU-Net v2", fontsize=11, fontweight="bold")
    if entropy is not None:
        axes[0, 3].set_title("nnU-Net + entropy selection", fontsize=11, fontweight="bold")
    figure.suptitle("Full-image spot check at matched locations", fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.015, wspace=0.025, hspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()

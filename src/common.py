"""Shared data loading and binary-segmentation metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class Roi:
    name: str
    image: np.ndarray
    mask: np.ndarray


def load_rois(annotation_dir: Path) -> list[Roi]:
    annotation_dir = annotation_dir.expanduser().resolve()
    image_paths = sorted(annotation_dir.glob("*-dapi.tif"))
    if len(image_paths) != 5:
        raise ValueError(f"Expected 5 ROI images in {annotation_dir}, found {len(image_paths)}")

    rois: list[Roi] = []
    for image_path in image_paths:
        mask_path = image_path.with_name(
            image_path.name.replace("-dapi.tif", "-dapi_mask.tif")
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}: {mask_path}")
        image = np.asarray(tifffile.imread(image_path))
        mask = np.asarray(tifffile.imread(mask_path)) > 0
        if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
            raise ValueError(
                f"ROI and mask must be matching 2-D arrays: {image_path.name}, "
                f"{image.shape} versus {mask.shape}"
            )
        rois.append(
            Roi(
                name=image_path.name.removesuffix("-dapi.tif"),
                image=image.astype(np.uint8, copy=False),
                mask=mask,
            )
        )
    return rois


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target {truth.shape}")

    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Dice": ratio(2 * tp, 2 * tp + fp + fn),
        "IoU": ratio(tp, tp + fp + fn),
        "FNR": ratio(fn, tp + fn),
        "FDR": ratio(fp, tp + fp),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }


def macro_average(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("At least one metric row is required")
    metrics = ("Dice", "IoU", "FNR", "FDR", "precision", "recall")
    result: dict[str, float | int] = {
        key: float(np.mean([float(row[key]) for row in rows])) for key in metrics
    }
    result["roi_count"] = len(rows)
    result["roi_meeting_both_10pct"] = sum(
        float(row["FNR"]) < 0.10 and float(row["FDR"]) < 0.10 for row in rows
    )
    return result


def pooled_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    counts = {key: sum(int(row[key]) for row in rows) for key in ("TP", "FP", "FN", "TN")}
    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        **counts,
        "Dice": ratio(2 * tp, 2 * tp + fp + fn),
        "IoU": ratio(tp, tp + fp + fn),
        "FNR": ratio(fn, tp + fn),
        "FDR": ratio(fp, tp + fp),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        (np.asarray(mask, dtype=bool) * 255).astype(np.uint8),
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "YX"},
    )

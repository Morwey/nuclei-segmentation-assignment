"""CPSAM-v2 zero-shot, leave-one-ROI-out fine-tuning, and final training."""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from cellpose import dynamics, models, train
from skimage import measure

from common import Roi, binary_metrics, load_rois, macro_average, save_binary_mask


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def degraded_copy(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape
    small = cv2.resize(
        image,
        (max(1, round(width / scale)), max(1, round(height / scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def proxy_instances(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels = measure.label(np.asarray(mask, dtype=bool), connectivity=1)
    keep = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            keep[labels == region.label] = True
    return measure.label(keep, connectivity=1).astype(np.int32)


def raw_prediction(
    model: models.CellposeModel, image: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    _, flows, _ = model.eval(
        image,
        diameter=float(config["diameter_px"]),
        flow_threshold=float(config["default_flow_threshold"]),
        cellprob_threshold=float(config["default_cellprob_threshold"]),
        min_size=int(config["min_size_px"]),
        normalize=True,
        compute_masks=False,
        batch_size=int(config["batch_size"]),
    )
    return np.asarray(flows[1], np.float32), np.asarray(flows[2], np.float32)


def labels_from_raw(
    raw: tuple[np.ndarray, np.ndarray],
    cellprob: float,
    flow: float,
    config: dict[str, Any],
) -> np.ndarray:
    d_p, probability = raw
    return np.asarray(
        dynamics.resize_and_compute_masks(
            d_p,
            probability,
            niter=200,
            cellprob_threshold=float(cellprob),
            flow_threshold=float(flow),
            min_size=int(config["min_size_px"]),
            max_size_fraction=0.4,
            device=torch.device("cpu"),
        ),
        dtype=np.int32,
    )


def threshold_grid(config: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(cellprob), float(flow))
        for cellprob in config["cellprob_grid"]
        for flow in config["flow_grid"]
    ]


def evaluate_grid(
    model: models.CellposeModel,
    images: list[np.ndarray],
    rois: list[Roi],
    config: dict[str, Any],
) -> tuple[dict[tuple[float, float], list[dict[str, float | int]]], list[tuple[np.ndarray, np.ndarray]]]:
    raw = [raw_prediction(model, image, config) for image in images]
    table: dict[tuple[float, float], list[dict[str, float | int]]] = {}
    for parameters in threshold_grid(config):
        table[parameters] = [
            binary_metrics(labels_from_raw(prediction, *parameters, config) > 0, roi.mask)
            for prediction, roi in zip(raw, rois)
        ]
    return table, raw


def select_threshold(
    table: dict[tuple[float, float], list[dict[str, float | int]]],
    indices: list[int],
    config: dict[str, Any],
) -> tuple[float, float]:
    margin = float(config["selection_error_margin"])

    def key(parameters: tuple[float, float]) -> tuple[float, float, float]:
        rows = [table[parameters][index] for index in indices]
        fnr = float(np.mean([row["FNR"] for row in rows]))
        fdr = float(np.mean([row["FDR"] for row in rows]))
        dice = float(np.mean([row["Dice"] for row in rows]))
        feasible = fnr <= margin and fdr <= margin
        return (1.0 if feasible else 0.0, dice if feasible else -max(fnr, fdr), -(fnr + fdr))

    return max(threshold_grid(config), key=key)


def conditions(rois: list[Roi], config: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    return {
        "native": [roi.image for roi in rois],
        "simulated_low_resolution": [
            degraded_copy(roi.image, float(config["resolution_scale"])) for roi in rois
        ],
    }


def save_fold_mask(
    output_dir: Path,
    roi: Roi,
    raw: tuple[np.ndarray, np.ndarray],
    parameters: tuple[float, float],
    config: dict[str, Any],
) -> None:
    save_binary_mask(
        output_dir / f"{roi.name}-dapi_mask.tif",
        labels_from_raw(raw, *parameters, config) > 0,
    )


def zero_shot(
    rois: list[Roi], device: torch.device, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    model = models.CellposeModel(device=device, pretrained_model=str(config["base_weights"]))
    parameters = (
        float(config["default_cellprob_threshold"]),
        float(config["default_flow_threshold"]),
    )
    result: dict[str, Any] = {}
    for name, images in conditions(rois, config).items():
        raw = [raw_prediction(model, image, config) for image in images]
        rows = [
            binary_metrics(labels_from_raw(prediction, *parameters, config) > 0, roi.mask)
            for prediction, roi in zip(raw, rois)
        ]
        result[name] = {"macro": macro_average(rows), "per_roi": rows}
        if name == "native":
            for roi, prediction in zip(rois, raw):
                save_fold_mask(output_dir, roi, prediction, parameters, config)
    del model
    gc.collect()
    return result


def training_data(
    rois: list[Roi], indices: list[int], config: dict[str, Any]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for index in indices:
        roi = rois[index]
        instance_mask = proxy_instances(roi.mask, int(config["min_size_px"]))
        images.append(roi.image)
        labels.append(instance_mask)
        if config["include_degraded_training_pairs"]:
            images.append(degraded_copy(roi.image, float(config["resolution_scale"])))
            labels.append(instance_mask.copy())
    return images, labels


def train_model(
    rois: list[Roi],
    indices: list[int],
    device: torch.device,
    config: dict[str, Any],
    work_dir: Path,
    model_name: str,
    seed: int,
) -> tuple[models.CellposeModel, Path]:
    set_seed(seed)
    model = models.CellposeModel(device=device, pretrained_model=str(config["base_weights"]))
    images, labels = training_data(rois, indices, config)
    model_path, _, _ = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        load_files=False,
        normalize=True,
        min_train_masks=0,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        n_epochs=int(config["epochs"]),
        nimg_per_epoch=int(config["images_per_epoch"]),
        scale_range=0.5,
        bsize=int(config["crop_size"]),
        save_path=work_dir,
        model_name=model_name,
    )
    del model
    gc.collect()
    return models.CellposeModel(device=device, pretrained_model=str(model_path)), Path(model_path)


def cross_validation(
    rois: list[Roi],
    device: torch.device,
    config: dict[str, Any],
    work_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for held_out, roi in enumerate(rois):
        train_indices = [index for index in range(len(rois)) if index != held_out]
        model, model_path = train_model(
            rois, train_indices, device, config, work_dir, f"cpsam_fold_{held_out}", seed + held_out
        )
        fold: dict[str, Any] = {"held_out": roi.name, "train_indices": train_indices, "conditions": {}}
        for condition_name, images in conditions(rois, config).items():
            table, raw = evaluate_grid(model, images, rois, config)
            parameters = select_threshold(table, train_indices, config)
            row = dict(table[parameters][held_out])
            row.update({"cellprob_threshold": parameters[0], "flow_threshold": parameters[1]})
            fold["conditions"][condition_name] = row
            if condition_name == "native":
                save_fold_mask(output_dir, roi, raw[held_out], parameters, config)
        folds.append(fold)
        del model
        model_path.unlink(missing_ok=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    return {
        "folds": folds,
        "native_macro": macro_average([fold["conditions"]["native"] for fold in folds]),
        "simulated_low_resolution_macro": macro_average(
            [fold["conditions"]["simulated_low_resolution"] for fold in folds]
        ),
    }


def train_all(
    rois: list[Roi],
    device: torch.device,
    config: dict[str, Any],
    work_dir: Path,
    output_model: Path,
    seed: int,
) -> dict[str, Any]:
    model, temporary_path = train_model(
        rois, list(range(len(rois))), device, config, work_dir, "cpsam_v2_all", seed
    )
    table, _ = evaluate_grid(model, [roi.image for roi in rois], rois, config)
    parameters = select_threshold(table, list(range(len(rois))), config)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_path, output_model)
    del model
    return {
        "model": str(output_model),
        "cellprob_threshold": parameters[0],
        "flow_threshold": parameters[1],
        "training_fit": macro_average(table[parameters]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("zero-shot", "cross-validation", "train-all"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--config", type=Path, default=Path("configs/cellpose_sam.json"))
    parser.add_argument("--output", type=Path, default=Path("results/cellpose_run.json"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/roi_predictions"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/cellpose"))
    parser.add_argument("--model-output", type=Path, default=Path("models/cpsam_v2_finetuned"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    rois = load_rois(args.annotations)
    device = select_device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "zero-shot":
        result = zero_shot(
            rois, device, config, args.prediction_dir / "cellpose_zero_shot"
        )
    elif args.mode == "cross-validation":
        result = cross_validation(
            rois,
            device,
            config,
            args.work_dir,
            args.prediction_dir / "cellpose_finetuned_loo",
            args.seed,
        )
    else:
        result = train_all(
            rois, device, config, args.work_dir, args.model_output, args.seed
        )
    payload = {"mode": args.mode, "device": str(device), "config": config, "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

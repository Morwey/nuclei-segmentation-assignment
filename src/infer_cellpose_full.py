"""Run tiled Cellpose-SAM inference on a full-resolution DAPI image."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from cellpose import models

from cellpose_pipeline import labels_from_raw, raw_prediction, read_config, select_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tile_origins(shape: tuple[int, int], tile_size: int) -> list[tuple[int, int]]:
    height, width = shape
    return [
        (x, y)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def infer_tile(
    model: models.CellposeModel,
    image: np.ndarray,
    x0: int,
    y0: int,
    tile_size: int,
    halo: int,
    scale: float,
    cellprob: float,
    flow: float,
    config: dict[str, Any],
) -> np.ndarray:
    height, width = image.shape
    x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
    ex0, ey0 = max(0, x0 - halo), max(0, y0 - halo)
    ex1, ey1 = min(width, x1 + halo), min(height, y1 + halo)
    extended = np.asarray(image[ey0:ey1, ex0:ex1])
    upsampled = cv2.resize(
        extended,
        (round(extended.shape[1] * scale), round(extended.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    raw = raw_prediction(model, upsampled, config)
    binary_high = labels_from_raw(raw, cellprob, flow, config) > 0
    binary_low = cv2.resize(
        binary_high.astype(np.uint8),
        (extended.shape[1], extended.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    cy0, cx0 = y0 - ey0, x0 - ex0
    return binary_low[cy0 : cy0 + (y1 - y0), cx0 : cx0 + (x1 - x0)]


def load_progress(path: Path, expected: dict[str, Any]) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Progress file does not match current argument: {key}")
    return int(payload["completed_tiles"])


def save_progress(path: Path, expected: dict[str, Any], completed: int) -> None:
    payload = dict(expected)
    payload["completed_tiles"] = completed
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/cellpose_sam.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scale", type=float, default=2.0445)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--halo", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cellprob", type=float, default=0.0)
    parser.add_argument("--flow", type=float, default=0.4)
    parser.add_argument("--work-mask", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--stop-after", type=int, help="Stop after this many tiles for timing tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = tifffile.memmap(args.image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, found {image.shape}")
    config = read_config(args.config)
    config["batch_size"] = args.batch_size
    device = select_device(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_mask = args.work_mask or args.output.with_name(args.output.stem + "_work.tif")
    progress = args.progress or args.output.with_name(args.output.stem + "_progress.json")
    work_mask.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    origins = tile_origins(tuple(image.shape), args.tile_size)
    expected = {
        "image": str(args.image.resolve()),
        "model": str(args.model.resolve()),
        "shape_yx": list(image.shape),
        "tile_size": args.tile_size,
        "halo": args.halo,
        "scale": args.scale,
        "cellprob_threshold": args.cellprob,
        "flow_threshold": args.flow,
    }
    completed = load_progress(progress, expected)
    if work_mask.exists():
        mask = tifffile.memmap(work_mask, mode="r+")
        if mask.shape != image.shape or mask.dtype != np.uint8:
            raise ValueError("Work mask shape or dtype does not match the input image")
    else:
        mask = tifffile.memmap(work_mask, shape=image.shape, dtype=np.uint8)
        mask[:] = 0
        mask.flush()

    print(f"Loading Cellpose-SAM model on {device}: {args.model}", flush=True)
    model = models.CellposeModel(device=device, pretrained_model=str(args.model))
    started = time.perf_counter()
    run_limit = len(origins) if args.stop_after is None else min(len(origins), completed + args.stop_after)
    for index in range(completed, run_limit):
        x0, y0 = origins[index]
        tile_started = time.perf_counter()
        prediction = infer_tile(
            model,
            image,
            x0,
            y0,
            args.tile_size,
            args.halo,
            args.scale,
            args.cellprob,
            args.flow,
            config,
        )
        mask[y0 : y0 + prediction.shape[0], x0 : x0 + prediction.shape[1]] = prediction * 255
        mask.flush()
        save_progress(progress, expected, index + 1)
        elapsed = time.perf_counter() - tile_started
        print(
            f"Tile {index + 1}/{len(origins)} x={x0} y={y0} | {elapsed:.1f}s",
            flush=True,
        )

    if run_limit < len(origins):
        print(f"Timing run stopped at tile {run_limit}/{len(origins)}", flush=True)
        return

    tifffile.imwrite(
        args.output,
        np.asarray(mask),
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "YX"},
    )
    saved = np.asarray(tifffile.imread(args.output))
    if saved.shape != image.shape or not set(np.unique(saved)).issubset({0, 255}):
        raise RuntimeError("Saved mask failed shape/value validation")
    metadata = {
        "file": args.output.name,
        "method": "Cellpose-SAM fine-tuned on all five ROIs",
        "model_sha256": sha256(args.model),
        "cellprob_threshold": args.cellprob,
        "flow_threshold": args.flow,
        "inference_scale": args.scale,
        "tile_size": args.tile_size,
        "halo": args.halo,
        "shape_yx": list(saved.shape),
        "dtype": str(saved.dtype),
        "values": [int(value) for value in np.unique(saved)],
        "foreground_fraction": float(np.mean(saved > 0)),
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        "inference_seconds": time.perf_counter() - started,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path, flush=True)


if __name__ == "__main__":
    main()

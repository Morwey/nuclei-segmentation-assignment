"""Run tiled full-image inference with the SPATCH entropy-selected nnU-Net model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tile_origins(shape: tuple[int, int], tile_size: int) -> list[tuple[int, int]]:
    height, width = shape
    return [
        (y, x)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--threshold", type=float, default=0.571)
    parser.add_argument("--scale", type=float, default=2.0445)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--work-mask", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = tifffile.memmap(args.image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Expected 2-D uint8 image, found {image.shape} {image.dtype}")
    height, width = image.shape
    origins = tile_origins((height, width), args.tile_size)
    signature = {
        "image": str(args.image.resolve()),
        "model": str(args.model.resolve()),
        "shape_yx": [height, width],
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "scale": args.scale,
        "tile_size": args.tile_size,
        "context": args.context,
    }

    args.work_mask.parent.mkdir(parents=True, exist_ok=True)
    if args.work_mask.exists():
        work_mask = tifffile.memmap(args.work_mask, mode="r+")
        progress = json.loads(args.progress.read_text(encoding="utf-8"))
        if progress["signature"] != signature:
            raise ValueError("Progress file does not match the requested inference")
        completed = int(progress["completed_tiles"])
    else:
        work_mask = tifffile.memmap(
            args.work_mask,
            shape=(height, width),
            dtype=np.uint8,
            bigtiff=True,
            photometric="minisblack",
        )
        work_mask[:] = 0
        work_mask.flush()
        completed = 0

    predictor = nnUNetPredictor(
        tile_step_size=1.0,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device(args.device),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model), (args.fold,), args.checkpoint
    )

    image_mean = float(np.mean(image, dtype=np.float64))
    image_std = float(np.std(image, dtype=np.float64))
    if image_std == 0:
        raise ValueError("Input image has zero standard deviation")
    started = time.time()

    for index, (y0, x0) in enumerate(origins[completed:], start=completed):
        y1 = min(y0 + args.tile_size, height)
        x1 = min(x0 + args.tile_size, width)
        ys = max(0, y0 - args.context)
        xs = max(0, x0 - args.context)
        ye = min(height, y1 + args.context)
        xe = min(width, x1 + args.context)
        source_tile = np.asarray(image[ys:ye, xs:xe])
        scaled = cv2.resize(
            source_tile,
            (
                round(source_tile.shape[1] * args.scale),
                round(source_tile.shape[0] * args.scale),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = (
            (scaled.astype(np.float32) - np.float32(image_mean))
            / np.float32(image_std)
        ).astype(np.float32, copy=False)
        logits = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(normalized[None, None])
        )
        probability = torch.softmax(logits.float(), dim=0)[1, 0].numpy()

        cy0 = round((y0 - ys) * args.scale)
        cx0 = round((x0 - xs) * args.scale)
        cy1 = round((y1 - ys) * args.scale)
        cx1 = round((x1 - xs) * args.scale)
        central = probability[cy0:cy1, cx0:cx1]
        native_probability = cv2.resize(
            central,
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_AREA,
        )
        work_mask[y0:y1, x0:x1] = (
            (native_probability >= args.threshold).astype(np.uint8) * 255
        )
        work_mask.flush()
        progress_payload = {
            "signature": signature,
            "completed_tiles": index + 1,
            "total_tiles": len(origins),
        }
        args.progress.parent.mkdir(parents=True, exist_ok=True)
        args.progress.write_text(
            json.dumps(progress_payload, indent=2) + "\n", encoding="utf-8"
        )
        if (index + 1) % 10 == 0 or index + 1 == len(origins):
            print(f"{index + 1}/{len(origins)} tiles", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        args.output,
        np.asarray(work_mask),
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "YX"},
    )
    result = np.asarray(tifffile.imread(args.output))
    if result.shape != image.shape or not set(np.unique(result)).issubset({0, 255}):
        raise RuntimeError("Full-image mask validation failed")
    metadata = {
        "file": args.output.name,
        "method": "nnU-Net v2 fold 0, SPATCH high-entropy 20, target/external 1:1",
        "model_dataset": "Dataset509_NucleiSPATCHEntropy20Balanced",
        "checkpoint": args.checkpoint,
        "probability_threshold": args.threshold,
        "inference_scale": args.scale,
        "tile_size": args.tile_size,
        "context": args.context,
        "normalization": "whole-image z-score",
        "shape_yx": [height, width],
        "dtype": str(result.dtype),
        "values": [int(value) for value in np.unique(result)],
        "foreground_fraction": float(np.mean(result > 0)),
        "tiles": len(origins),
        "elapsed_seconds": time.time() - started,
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Guarded, local slide-restoration pipeline for Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(os.environ.get("SLIDE_PIPELINE_ROOT", Path(__file__).parent)).resolve()
STATE = ROOT / ".slide_pipeline"
VENDOR = STATE / "vendor" / "seedvr2"
MODELS = STATE / "models"
ORIGINAL_DIRS = ((ROOT / "manual").resolve(), (ROOT / "machine").resolve())
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_paths(input_dir: Path, output_dir: Path, work_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    work_dir = work_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    for original in ORIGINAL_DIRS:
        if is_within(input_dir, original):
            raise SystemExit(
                f"Refusing to process originals: {input_dir}. Copy images to a separate directory first."
            )
        if is_within(output_dir, original) or is_within(work_dir, original):
            raise SystemExit(f"Refusing to write inside originals directory: {original}")
    if input_dir == output_dir or is_within(output_dir, input_dir):
        raise SystemExit("Output must be outside the input directory.")
    if input_dir == work_dir or is_within(work_dir, input_dir):
        raise SystemExit("Work directory must be outside the input directory.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_color_and_tone(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Conservative, image-adaptive cast and contrast correction."""
    image = rgb.astype(np.float32) / 255.0
    linear = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    maximum = linear.max(axis=2)
    minimum = linear.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    lo, hi = np.percentile(luminance, [15.0, 92.0])
    neutral = (saturation < 0.20) & (luminance >= lo) & (luminance <= hi)
    if neutral.sum() < max(500, image.shape[0] * image.shape[1] * 0.002):
        neutral = (saturation < 0.38) & (luminance >= lo) & (luminance <= hi)
    samples = linear[neutral] if neutral.any() else linear.reshape(-1, 3)
    channel_level = np.median(samples, axis=0)
    target = float(np.exp(np.mean(np.log(np.maximum(channel_level, 1e-6)))))
    gains = np.clip(target / np.maximum(channel_level, 1e-6), 0.78, 1.28)
    cast_strength = float(np.max(np.abs(gains - 1.0)))
    correction_blend = float(np.clip(0.45 + cast_strength, 0.45, 0.72))
    balanced = linear * (1.0 + correction_blend * (gains - 1.0))[None, None, :]
    balanced = np.clip(balanced, 0.0, 1.0)
    srgb = np.where(
        balanced <= 0.0031308,
        balanced * 12.92,
        1.055 * np.power(balanced, 1.0 / 2.4) - 0.055,
    )
    corrected = np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    enhanced_l = cv2.addWeighted(lightness, 0.42, enhanced_l, 0.58, 0)
    corrected = cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2RGB)
    metrics = {
        "red_gain": float(gains[0]),
        "green_gain": float(gains[1]),
        "blue_gain": float(gains[2]),
        "cast_strength": cast_strength,
    }
    return corrected, metrics


def conservative_sharpen(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    amount = float(np.interp(np.clip(focus_score, 40.0, 600.0), [40.0, 600.0], [0.46, 0.16]))
    blurred = cv2.GaussianBlur(rgb, (0, 0), 0.85)
    sharpened = cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0)
    difference = cv2.absdiff(rgb, blurred).max(axis=2)
    result = rgb.copy()
    mask = difference >= 3
    result[mask] = sharpened[mask]
    return result, focus_score


def prepare_image(source: Path, destination: Path) -> dict[str, object]:
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = np.asarray(oriented)
    corrected, color_metrics = robust_color_and_tone(rgb)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected, "RGB").save(destination, format="PNG", compress_level=2)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "prepared": str(destination),
        "input_width": int(rgb.shape[1]),
        "input_height": int(rgb.shape[0]),
        **color_metrics,
    }


def finish_image(restored: Path, master: Path, delivery: Path) -> dict[str, object]:
    with Image.open(restored) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    sharpened, focus_score = conservative_sharpen(rgb)
    master.parent.mkdir(parents=True, exist_ok=True)
    delivery.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sharpened, "RGB").save(master, format="PNG", compress_level=4)
    Image.fromarray(sharpened, "RGB").save(
        delivery, format="JPEG", quality=95, subsampling=0, optimize=True
    )
    return {
        "master": str(master),
        "delivery": str(delivery),
        "output_width": int(rgb.shape[1]),
        "output_height": int(rgb.shape[0]),
        "focus_score": focus_score,
        "master_sha256": sha256(master),
    }


def doctor() -> None:
    import torch

    problems: list[str] = []
    if sys.platform != "darwin":
        problems.append("host is not macOS")
    if not torch.backends.mps.is_available():
        problems.append("PyTorch MPS is unavailable")
    if not (VENDOR / "inference_cli.py").is_file():
        problems.append("SeedVR2 checkout is missing")
    for model in (DIT_MODEL, VAE_MODEL):
        if not (MODELS / model).is_file():
            problems.append(f"model is missing: {model}")
    if problems:
        raise SystemExit("Doctor failed: " + "; ".join(problems))
    print(
        json.dumps(
            {
                "status": "ok",
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "mps_available": True,
                "seedvr2_commit": subprocess.check_output(
                    ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True
                ).strip(),
                "models": [DIT_MODEL, VAE_MODEL],
            },
            indent=2,
        )
    )


def collect_images(input_dir: Path, recursive: bool, limit: int | None) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")
    return images


def run(args: argparse.Namespace) -> None:
    if args.resolution_quantum <= 0:
        raise SystemExit("--resolution-quantum must be a positive integer")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    guard_paths(input_dir, output_dir, work_dir)
    doctor()
    images = collect_images(input_dir, args.recursive, args.limit)

    prepared_dir = work_dir / "prepared"
    seedvr_dir = work_dir / "seedvr2"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "pipeline": "slide-restoration-macos-v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [],
    }

    buckets: dict[int, list[dict[str, object]]] = {}
    print(f"Preparing {len(images)} copied input images")
    for source in images:
        resolved_source = source.resolve()
        for original in ORIGINAL_DIRS:
            if is_within(resolved_source, original):
                raise SystemExit(
                    f"Refusing an input that resolves into originals: {source} -> {resolved_source}"
                )
        relative = source.relative_to(input_dir)
        with Image.open(source) as opened:
            oriented_size = ImageOps.exif_transpose(opened).size
        native_short_edge = min(oriented_size)
        model_resolution = max(
            1024,
            int(math.ceil(native_short_edge / args.resolution_quantum) * args.resolution_quantum),
        )
        unique_name = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:10]
        prepared_name = f"{unique_name}__{source.stem}.png"
        prepared = prepared_dir / str(model_resolution) / prepared_name
        record = prepare_image(source, prepared)
        record["relative"] = str(relative)
        record["prepared_name"] = prepared_name
        record["native_short_edge"] = native_short_edge
        record["model_resolution"] = model_resolution
        manifest["files"].append(record)
        buckets.setdefault(model_resolution, []).append(record)

    print(f"Running SeedVR2 restoration in {len(buckets)} native-resolution bucket(s)")
    for model_resolution in sorted(buckets):
        bucket_input = prepared_dir / str(model_resolution)
        bucket_output = seedvr_dir / str(model_resolution)
        seed_command = [
            sys.executable,
            str(VENDOR / "inference_cli.py"),
            str(bucket_input),
            "--output",
            str(bucket_output),
            "--output_format",
            "png",
            "--dit_model",
            DIT_MODEL,
            "--model_dir",
            str(MODELS),
            "--resolution",
            str(model_resolution),
            "--batch_size",
            "1",
            "--seed",
            str(args.seed),
            "--color_correction",
            "lab",
            "--vae_encode_tiled",
            "--vae_decode_tiled",
            "--cache_dit",
            "--cache_vae",
        ]
        subprocess.run(seed_command, cwd=VENDOR, check=True)

    records = manifest["files"]
    assert isinstance(records, list)
    print("Writing restored masters and delivery JPEGs")
    for record in records:
        assert isinstance(record, dict)
        relative = Path(str(record["relative"]))
        restored = (
            seedvr_dir
            / str(record["model_resolution"])
            / str(record["prepared_name"])
        )
        if not restored.is_file():
            raise SystemExit(f"SeedVR2 did not produce expected output: {restored}")
        master = (output_dir / "masters" / relative).with_suffix(".png")
        delivery = (output_dir / "delivery" / relative).with_suffix(".jpg")
        record.update(finish_image(restored, master, delivery))

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path = output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(manifest_path)
    print(f"Completed {len(images)} images; manifest: {manifest_path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="verify the complete local installation")
    run_parser = subparsers.add_parser("run", help="restore images from a copied input directory")
    run_parser.add_argument("--input-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--work-dir", type=Path, required=True)
    run_parser.add_argument("--recursive", action="store_true")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--resolution-quantum",
        type=int,
        default=256,
        help="round each native short edge upward to this bucket size; never downsamples",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "doctor":
        doctor()
    else:
        run(args)


if __name__ == "__main__":
    main()

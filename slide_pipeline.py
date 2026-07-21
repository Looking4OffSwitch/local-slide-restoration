#!/usr/bin/env python3
"""Guarded, local slide-restoration pipeline for Apple Silicon and Bazzite."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

ROOT = Path(os.environ.get("SLIDE_PIPELINE_ROOT", Path(__file__).parent)).resolve()
STATE = ROOT / ".slide_pipeline"
VENDOR = STATE / "vendor" / "seedvr2"
MODELS = STATE / "models"
ORIGINAL_DIRS = tuple(
    (ROOT / directory).resolve() for directory in ("originals", "manual", "machine")
)
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
FP16_DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
FP8_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PREPARATION_VERSION = "color-tone-v2-icc-srgb"
MANIFEST_SCHEMA_VERSION = 4
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_PROFILE_BYTES = SRGB_PROFILE.tobytes()


class Progress(Protocol):
    n: int | float

    def update(self, value: int | float) -> object: ...


@dataclass(frozen=True)
class RestorationProfile:
    name: str
    model: str
    description: str
    cuda_blocks_to_swap: int


PROFILES = {
    "archival-fp16": RestorationProfile(
        name="archival-fp16",
        model=FP16_DIT_MODEL,
        description="FP16 archival-quality model with maximum CUDA BlockSwap",
        cuda_blocks_to_swap=32,
    ),
    "balanced-fp8": RestorationProfile(
        name="balanced-fp8",
        model=FP8_DIT_MODEL,
        description="FP8 model recommended by SeedVR2 for 12-16 GB CUDA GPUs",
        cuda_blocks_to_swap=16,
    ),
}
DEFAULT_PROFILE = "archival-fp16"


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
        if is_within(output_dir, original) or is_within(work_dir, original):
            raise SystemExit(f"Refusing to write inside originals directory: {original}")
    if input_dir == output_dir or is_within(output_dir, input_dir):
        raise SystemExit("Output must be outside the input directory.")
    if input_dir == work_dir or is_within(work_dir, input_dir):
        raise SystemExit("Work directory must be outside the input directory.")
    if output_dir == work_dir or is_within(output_dir, work_dir) or is_within(work_dir, output_dir):
        raise SystemExit("Output and work directories must be separate.")


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


def color_managed_rgb(opened: Image.Image) -> tuple[Image.Image, dict[str, object]]:
    """Convert tagged input to sRGB and document the assumption for untagged input."""
    embedded = opened.info.get("icc_profile")
    if embedded:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
            converted = ImageCms.profileToProfile(
                opened, source_profile, SRGB_PROFILE, outputMode="RGB"
            )
        except (OSError, ImageCms.PyCMSError) as error:
            raise ValueError(f"invalid embedded ICC profile: {error}") from error
        profile_source = "embedded"
        profile_sha256 = hashlib.sha256(embedded).hexdigest()
    else:
        converted = opened.convert("RGB")
        profile_source = "assumed-srgb"
        profile_sha256 = None
    if converted is None:
        raise ValueError("ICC conversion did not produce an image")
    return converted, {
        "input_icc_source": profile_source,
        "input_icc_sha256": profile_sha256,
        "working_color_space": "sRGB IEC61966-2.1",
        "output_icc_sha256": hashlib.sha256(SRGB_PROFILE_BYTES).hexdigest(),
    }


def atomic_save(image: Image.Image, destination: Path, **save_options: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        image.save(temporary_path, **save_options)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_image(source: Path, destination: Path) -> dict[str, object]:
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        converted, color_profile = color_managed_rgb(oriented)
        converted.load()
        rgb = np.asarray(converted)
    corrected, color_metrics = robust_color_and_tone(rgb)
    atomic_save(
        Image.fromarray(corrected, "RGB"),
        destination,
        format="PNG",
        compress_level=2,
        icc_profile=SRGB_PROFILE_BYTES,
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "prepared": str(destination),
        "input_width": int(rgb.shape[1]),
        "input_height": int(rgb.shape[0]),
        "prepared_sha256": sha256(destination),
        "preparation_version": PREPARATION_VERSION,
        **color_profile,
        **color_metrics,
    }


def finish_image(
    seedvr_output: Path,
    original: Path,
    restored: Path,
    output_size: tuple[int, int],
) -> dict[str, object]:
    with Image.open(seedvr_output) as opened:
        converted = opened.convert("RGB")
        model_output_size = converted.size
        if converted.size != output_size:
            converted = converted.resize(output_size, Image.Resampling.LANCZOS)
        rgb = np.asarray(converted)
    sharpened, focus_score = conservative_sharpen(rgb)
    result = Image.fromarray(sharpened, "RGB")
    atomic_save(
        result,
        original,
        format="PNG",
        compress_level=4,
        icc_profile=SRGB_PROFILE_BYTES,
    )
    atomic_save(
        result,
        restored,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
        icc_profile=SRGB_PROFILE_BYTES,
    )
    return {
        "original": str(original),
        "restored": str(restored),
        "output_width": int(rgb.shape[1]),
        "output_height": int(rgb.shape[0]),
        "model_output_width": model_output_size[0],
        "model_output_height": model_output_size[1],
        "focus_score": focus_score,
        "original_sha256": sha256(original),
        "restored_sha256": sha256(restored),
        "output_color_space": "sRGB IEC61966-2.1",
        "output_bit_depth": 8,
    }


def read_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    os_release = Path("/etc/os-release")
    if not os_release.is_file():
        return values
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def detect_backend(torch_module: Any) -> str | None:
    if sys.platform == "darwin" and torch_module.backends.mps.is_available():
        return "mps"
    if sys.platform.startswith("linux") and torch_module.cuda.is_available():
        return "cuda"
    return None


def doctor(profile_names: list[str]) -> dict[str, object]:
    import psutil
    import torch

    problems: list[str] = []
    backend = detect_backend(torch)
    system = platform.system()
    machine = platform.machine()
    os_release = read_os_release()
    system_ram_gib = psutil.virtual_memory().total / (1024**3)
    cuda_devices: list[dict[str, object]] = []
    if system == "Darwin":
        if machine != "arm64":
            problems.append("macOS host is not Apple Silicon")
        if backend != "mps":
            problems.append("PyTorch MPS is unavailable")
    elif system == "Linux":
        if machine != "x86_64":
            problems.append(f"Bazzite host architecture is unsupported: {machine}")
        if os_release.get("ID") != "bazzite":
            problems.append(
                f"Linux host is not Bazzite (reported ID={os_release.get('ID', 'unknown')})"
            )
        if backend != "cuda":
            problems.append("PyTorch CUDA is unavailable")
        if shutil.which("nvidia-smi") is None:
            problems.append("nvidia-smi is unavailable")
        if backend == "cuda":
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda_devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "vram_gib": round(properties.total_memory / (1024**3), 2),
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )
            if not cuda_devices:
                problems.append("CUDA reports no NVIDIA devices")
            else:
                selected_device = cuda_devices[0]
                if "RTX 4080" not in str(selected_device["name"]):
                    problems.append(
                        f"CUDA device 0 is not the expected RTX 4080: {selected_device['name']}"
                    )
                selected_vram_gib = float(str(selected_device["vram_gib"]))
                if not 11.0 <= selected_vram_gib <= 13.0:
                    problems.append(
                        f"CUDA device 0 reports {selected_vram_gib} GiB, "
                        "not the expected 12 GB VRAM"
                    )
            if system_ram_gib < 60.0:
                problems.append("host reports less than the expected 64 GB system RAM")
    else:
        problems.append(f"unsupported operating system: {system}")
    if not (VENDOR / "inference_cli.py").is_file():
        problems.append("SeedVR2 checkout is missing")
    seedvr2_commit = None
    if (VENDOR / ".git").is_dir():
        commit_result = subprocess.run(
            ["git", "-C", str(VENDOR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0:
            problems.append("SeedVR2 checkout commit cannot be read")
        else:
            seedvr2_commit = commit_result.stdout.strip()
        if seedvr2_commit is not None and seedvr2_commit != SEEDVR2_COMMIT:
            problems.append(f"SeedVR2 checkout is at {seedvr2_commit}, expected {SEEDVR2_COMMIT}")
    else:
        problems.append("SeedVR2 Git metadata is missing")
    requested_models = [PROFILES[name].model for name in profile_names]
    for model in (*requested_models, VAE_MODEL):
        if not (MODELS / model).is_file():
            problems.append(f"model is missing: {model}")
    if problems:
        raise SystemExit("Doctor failed: " + "; ".join(problems))
    report: dict[str, object] = {
        "status": "ok",
        "os": system,
        "os_id": os_release.get("ID") if os_release else None,
        "architecture": machine,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "backend": backend,
        "system_ram_gib": round(system_ram_gib, 2),
        "seedvr2_commit": seedvr2_commit,
        "profiles": profile_names,
        "models": [*requested_models, VAE_MODEL],
    }
    if backend == "cuda":
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_devices"] = cuda_devices
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["nvidia_driver"] = nvidia_smi.stdout.splitlines()[0].strip()
    else:
        report["mps_available"] = True
    print(json.dumps(report, indent=2))
    return report


def seedvr_arguments(
    profile: RestorationProfile, backend: str, cuda_blocks_to_swap: int | None
) -> list[str]:
    arguments = [
        "--dit_model",
        profile.model,
        "--vae_encode_tiled",
        "--vae_decode_tiled",
        "--cache_dit",
        "--cache_vae",
    ]
    if backend == "cuda":
        blocks_to_swap = (
            profile.cuda_blocks_to_swap if cuda_blocks_to_swap is None else cuda_blocks_to_swap
        )
        arguments.extend(
            [
                "--cuda_device",
                "0",
                "--dit_offload_device",
                "cpu",
                "--vae_offload_device",
                "cpu",
                "--tensor_offload_device",
                "cpu",
                "--blocks_to_swap",
                str(blocks_to_swap),
            ]
        )
        if blocks_to_swap > 0:
            arguments.append("--swap_io_components")
    return arguments


def collect_images(input_dir: Path, recursive: bool, limit: int | None) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")
    return images


def default_work_dir(output_dir: Path) -> Path:
    """Keep intermediates beside, but separate from, the result directory."""
    return output_dir.parent / f".{output_dir.name}-work"


def output_paths(output_dir: Path, relative: Path) -> tuple[Path, Path]:
    """Preserve the source extension so unlike source formats cannot collide."""
    original = output_dir / "originals" / relative.parent / f"{relative.name}.png"
    restored = output_dir / "restored" / relative.parent / f"{relative.name}.jpg"
    return original, restored


def installed_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "opencv-python",
        "pillow",
        "scikit-image",
        "torch",
        "torchvision",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def validate_output_targets(
    output_dir: Path, input_dir: Path, images: list[Path], overwrite: bool
) -> None:
    targets: list[Path] = []
    for source in images:
        targets.extend(output_paths(output_dir, source.relative_to(input_dir)))
    if len(targets) != len(set(targets)):
        raise SystemExit("Multiple inputs resolve to the same final output path.")
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {len(existing)} existing output file(s); "
            "use --overwrite after reviewing the destination."
        )


def load_preparation_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != 2:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {key: value for key, value in entries.items() if isinstance(value, dict)}


def save_preparation_cache(cache_path: Path, entries: dict[str, dict[str, object]]) -> None:
    """Atomically checkpoint preparation after each image."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=cache_path.parent, delete=False, encoding="utf-8"
    ) as temporary:
        json.dump({"version": 2, "entries": entries}, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(cache_path)


def cached_preparation(
    source: Path,
    prepared: Path,
    cache: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    entry = cache.get(str(source.resolve()))
    if entry is None or not prepared.is_file():
        return None
    stat = source.stat()
    if (
        entry.get("source_size") != stat.st_size
        or entry.get("source_mtime_ns") != stat.st_mtime_ns
        or entry.get("prepared") != str(prepared)
        or entry.get("preparation_version") != PREPARATION_VERSION
        or entry.get("source_sha256") != sha256(source)
        or entry.get("prepared_sha256") != sha256(prepared)
    ):
        return None
    return dict(entry)


def run_seedvr_with_progress(
    command: list[str], bucket_size: int, completed_before: int, progress: Progress
) -> None:
    """Stream SeedVR2 output while deriving whole-run progress from its file counter."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=VENDOR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("SeedVR2 output stream was not created")
    for line in process.stdout:
        print(line, end="", flush=True)
        match = re.search(r"Processing file (\d+)/(\d+)", line)
        if match is not None:
            current = int(match.group(1))
            reported_total = int(match.group(2))
            if reported_total == bucket_size:
                target = completed_before + current - 1
                progress.update(max(0, target - progress.n))
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    target = completed_before + bucket_size
    progress.update(max(0, target - progress.n))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.resolution_quantum <= 0:
        raise SystemExit("--resolution-quantum must be a positive integer")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    if args.cuda_blocks_to_swap is not None and not 0 <= args.cuda_blocks_to_swap <= 32:
        raise SystemExit("--cuda-blocks-to-swap must be between 0 and 32")
    input_image_arg = getattr(args, "input_image", None)
    if input_image_arg is not None:
        input_image = input_image_arg.resolve()
        if not input_image.is_file() or input_image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"Input is not a supported image: {input_image}")
        input_dir = input_image.parent
        images = [input_image]
    else:
        input_image = None
        input_dir_arg = getattr(args, "input_dir", None)
        if input_dir_arg is None:
            raise SystemExit("One of --input-image or --input-dir is required")
        input_dir = input_dir_arg.resolve()
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --simple is used")
    output_dir = args.output_dir.resolve()
    work_dir = (
        args.work_dir.resolve() if args.work_dir is not None else default_work_dir(output_dir)
    )
    guard_paths(input_dir, output_dir, work_dir)
    if input_image is None:
        images = collect_images(input_dir, args.recursive, args.limit)
    profile = PROFILES[args.profile]
    installation = doctor([profile.name])
    backend = str(installation["backend"])
    if backend != "cuda" and args.cuda_blocks_to_swap is not None:
        raise SystemExit("--cuda-blocks-to-swap is only valid on the Bazzite CUDA host")
    validate_output_targets(output_dir, input_dir, images, bool(getattr(args, "overwrite", False)))

    prepared_dir = work_dir / "prepared"
    seedvr_dir = work_dir / "seedvr2"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "pipeline": "local-slide-restoration-v3",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": backend,
        "profile": profile.name,
        "profile_description": profile.description,
        "dit_model": profile.model,
        "cuda_blocks_to_swap": (
            profile.cuda_blocks_to_swap
            if backend == "cuda" and args.cuda_blocks_to_swap is None
            else args.cuda_blocks_to_swap
        ),
        "input_dir": str(input_dir),
        "input_image": str(input_image) if input_image is not None else None,
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "software_versions": installed_versions(),
        "preparation_version": PREPARATION_VERSION,
        "files": manifest_records,
    }

    buckets: dict[int, list[dict[str, object]]] = {}
    from tqdm.auto import tqdm

    for source in tqdm(images, desc="Scanning image dimensions", unit="photo", dynamic_ncols=True):
        relative = source.relative_to(input_dir)
        try:
            with Image.open(source) as opened:
                oriented_size = ImageOps.exif_transpose(opened).size
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot read input image {source}: {error}") from error
        native_short_edge = min(oriented_size)
        model_resolution = max(
            1024,
            int(math.ceil(native_short_edge / args.resolution_quantum) * args.resolution_quantum),
        )
        unique_name = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:10]
        prepared_name = f"{unique_name}__{source.stem}.png"
        prepared = prepared_dir / str(model_resolution) / prepared_name
        buckets.setdefault(model_resolution, []).append(
            {
                "source_path": source,
                "relative": str(relative),
                "prepared_name": prepared_name,
                "prepared_path": prepared,
                "native_short_edge": native_short_edge,
                "model_resolution": model_resolution,
            }
        )

    cache_path = work_dir / "preparation-cache.json"
    preparation_cache = load_preparation_cache(cache_path)
    print(f"Processing {len(images)} images in {len(buckets)} native-resolution bucket(s)")
    restored_count = 0
    with tqdm(
        total=len(images),
        desc="Restoring",
        unit="photo",
        dynamic_ncols=True,
    ) as restoration_progress:
        for model_resolution in sorted(buckets):
            bucket = buckets[model_resolution]
            records: list[dict[str, object]] = []
            cached_count = 0
            with tqdm(
                bucket,
                desc=f"Color correcting and converting ({model_resolution}px)",
                unit="photo",
                dynamic_ncols=True,
            ) as preparation_progress:
                for item in preparation_progress:
                    source = item["source_path"]
                    prepared = item["prepared_path"]
                    if not isinstance(source, Path) or not isinstance(prepared, Path):
                        raise RuntimeError("Internal preparation record has invalid paths")
                    record = cached_preparation(source, prepared, preparation_cache)
                    if record is not None:
                        cached_count += 1
                    else:
                        record = prepare_image(source, prepared)
                        source_stat = source.stat()
                        record["source_size"] = source_stat.st_size
                        record["source_mtime_ns"] = source_stat.st_mtime_ns
                        preparation_cache[str(source.resolve())] = dict(record)
                        save_preparation_cache(cache_path, preparation_cache)
                    preparation_progress.set_postfix_str(f"{cached_count} cached", refresh=False)
                    record["relative"] = item["relative"]
                    record["prepared_name"] = item["prepared_name"]
                    record["native_short_edge"] = item["native_short_edge"]
                    record["model_resolution"] = model_resolution
                    records.append(record)
                    manifest_records.append(record)

            bucket_output = seedvr_dir / str(model_resolution)
            bucket_output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"seedvr-input-{model_resolution}-", dir=work_dir
            ) as temporary_bucket:
                bucket_input = Path(temporary_bucket)
                for record in records:
                    prepared = Path(str(record["prepared"]))
                    os.link(prepared, bucket_input / str(record["prepared_name"]))
                seed_command = [
                    sys.executable,
                    str(VENDOR / "inference_cli.py"),
                    str(bucket_input),
                    "--output",
                    str(bucket_output),
                    "--output_format",
                    "png",
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
                    *seedvr_arguments(profile, backend, args.cuda_blocks_to_swap),
                ]
                bucket_size = len(bucket)
                run_seedvr_with_progress(
                    seed_command,
                    bucket_size,
                    restored_count,
                    restoration_progress,
                )
            restored_count += bucket_size

    for record in tqdm(manifest_records, desc="Finalizing", unit="photo", dynamic_ncols=True):
        relative = Path(str(record["relative"]))
        restored = seedvr_dir / str(record["model_resolution"]) / str(record["prepared_name"])
        if not restored.is_file():
            raise SystemExit(f"SeedVR2 did not produce expected output: {restored}")
        original, restored_output = output_paths(output_dir, relative)
        output_size = (
            int(str(record["input_width"])),
            int(str(record["input_height"])),
        )
        record.update(finish_image(restored, original, restored_output, output_size))

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path = output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(manifest_path)
    print(f"Completed {len(images)} images; manifest: {manifest_path}")
    return manifest


def simple_destination(source: Path, current_dir: Path | None = None) -> Path:
    output_dir = Path.cwd() if current_dir is None else current_dir
    destination = (output_dir / f"{source.stem}_restored{source.suffix}").resolve()
    for original_dir in ORIGINAL_DIRS:
        if is_within(destination, original_dir):
            raise SystemExit(
                f"Refusing to write inside protected originals directory: {original_dir}"
            )
    return destination


def confirm_simple_overwrite(destination: Path) -> bool:
    if not destination.exists():
        return True
    try:
        answer = input(f"{destination} already exists. Overwrite it? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def simple(args: argparse.Namespace) -> Path:
    if args.input_image is None:
        raise SystemExit("--simple requires --input-image")
    if args.output_dir is not None or args.work_dir is not None:
        raise SystemExit("--simple cannot be combined with --output-dir or --work-dir")
    if args.limit is not None:
        raise SystemExit("--simple cannot be combined with --limit")
    source = args.input_image.resolve()
    if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Input is not a supported image: {source}")
    destination = simple_destination(source)
    if not confirm_simple_overwrite(destination):
        raise SystemExit("Restoration cancelled; existing output was not changed.")

    with tempfile.TemporaryDirectory(prefix="slide-restoration-simple-") as temporary:
        temporary_root = Path(temporary)
        run_args = argparse.Namespace(
            input_image=source,
            input_dir=None,
            output_dir=temporary_root / "output",
            work_dir=temporary_root / "work",
            recursive=False,
            limit=None,
            seed=args.seed,
            resolution_quantum=args.resolution_quantum,
            profile=args.profile,
            cuda_blocks_to_swap=args.cuda_blocks_to_swap,
            overwrite=True,
        )
        manifest = run(run_args)
        records = manifest["files"]
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise RuntimeError("Simple restoration did not return exactly one file record")
        record = cast(dict[str, object], records[0])
        png_result = Path(str(record["original"]))
        jpeg_result = Path(str(record["restored"]))
        suffix = source.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            atomic_copy(jpeg_result, destination)
        elif suffix == ".png":
            atomic_copy(png_result, destination)
        else:
            with Image.open(png_result) as opened:
                result = opened.convert("RGB")
                result.load()
            if suffix in {".tif", ".tiff"}:
                atomic_save(
                    result,
                    destination,
                    format="TIFF",
                    compression="tiff_lzw",
                    icc_profile=SRGB_PROFILE_BYTES,
                )
            else:
                atomic_save(
                    result,
                    destination,
                    format="WEBP",
                    quality=95,
                    method=6,
                    icc_profile=SRGB_PROFILE_BYTES,
                )
    print(f"Restored image: {destination}")
    return destination


def compare_profile_outputs(fp16_original: Path, fp8_original: Path) -> dict[str, float | None]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    with Image.open(fp16_original) as opened:
        fp16 = np.asarray(opened.convert("RGB"))
    with Image.open(fp8_original) as opened:
        fp8 = np.asarray(opened.convert("RGB"))
    if fp16.shape != fp8.shape:
        raise SystemExit(
            f"Benchmark outputs have different dimensions: {fp16.shape} versus {fp8.shape}"
        )
    absolute_difference = np.abs(fp16.astype(np.int16) - fp8.astype(np.int16))
    maximum_difference = float(absolute_difference.max())
    psnr = (
        None
        if maximum_difference == 0.0
        else float(peak_signal_noise_ratio(fp16, fp8, data_range=255))
    )
    return {
        "mean_absolute_channel_difference": float(absolute_difference.mean()),
        "maximum_channel_difference": maximum_difference,
        "psnr_db": psnr,
        "ssim": float(structural_similarity(fp16, fp8, channel_axis=2, data_range=255)),
    }


def benchmark(args: argparse.Namespace) -> None:
    if args.resolution_quantum <= 0:
        raise SystemExit("--resolution-quantum must be a positive integer")
    source = args.input_image.resolve()
    if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Benchmark input is not a supported image: {source}")
    for original in ORIGINAL_DIRS:
        if is_within(source, original):
            raise SystemExit(
                f"Refusing to benchmark an original: {source}. Copy it elsewhere first."
            )
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve()
    if output_dir == work_dir or is_within(output_dir, work_dir) or is_within(work_dir, output_dir):
        raise SystemExit("Benchmark output and work directories must be separate.")
    if is_within(source, output_dir) or is_within(source, work_dir):
        raise SystemExit("Benchmark input must be outside the output and work directories.")
    installation = doctor(list(PROFILES))
    if installation["backend"] != "cuda":
        raise SystemExit("The FP16-versus-FP8 benchmark requires the Bazzite CUDA host.")

    copied_input_dir = work_dir / "copied-input"
    copied_input_dir.mkdir(parents=True, exist_ok=True)
    copied_input = copied_input_dir / source.name
    shutil.copy2(source, copied_input)
    if sha256(source) != sha256(copied_input):
        raise SystemExit("Benchmark input copy failed SHA-256 verification.")

    profile_results: dict[str, object] = {}
    results: dict[str, object] = {
        "pipeline": "local-slide-restoration-profile-benchmark-v1",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(source),
        "source_sha256": sha256(source),
        "seed": args.seed,
        "resolution_quantum": args.resolution_quantum,
        "execution_order": ["archival-fp16", "balanced-fp8"],
        "environment": installation,
        "profiles": profile_results,
    }
    originals: dict[str, Path] = {}
    failures: list[str] = []
    for profile_name in ("archival-fp16", "balanced-fp8"):
        profile_output = output_dir / profile_name
        profile_work = work_dir / profile_name
        run_args = argparse.Namespace(
            input_dir=copied_input_dir,
            output_dir=profile_output,
            work_dir=profile_work,
            recursive=False,
            limit=None,
            seed=args.seed,
            resolution_quantum=args.resolution_quantum,
            profile=profile_name,
            cuda_blocks_to_swap=None,
            overwrite=True,
        )
        started = time.perf_counter()
        try:
            manifest = run(run_args)
        except subprocess.CalledProcessError as error:
            elapsed_seconds = time.perf_counter() - started
            profile_results[profile_name] = {
                "status": "failed",
                "elapsed_seconds": elapsed_seconds,
                "return_code": error.returncode,
                "command": [str(part) for part in error.cmd],
            }
            failures.append(profile_name)
            continue
        elapsed_seconds = time.perf_counter() - started
        records = manifest["files"]
        if not isinstance(records, list) or len(records) != 1:
            raise RuntimeError("Benchmark run did not return exactly one file record")
        record = records[0]
        if not isinstance(record, dict):
            raise RuntimeError("Benchmark file record is invalid")
        record = cast(dict[str, object], record)
        original = Path(str(record["original"]))
        originals[profile_name] = original
        profile_results[profile_name] = {
            "status": "ok",
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "model": manifest["dit_model"],
            "cuda_blocks_to_swap": manifest["cuda_blocks_to_swap"],
            "original": str(original),
            "original_sha256": record["original_sha256"],
            "output_width": record["output_width"],
            "output_height": record["output_height"],
        }

    if not failures:
        results["output_comparison"] = compare_profile_outputs(
            originals["archival-fp16"], originals["balanced-fp8"]
        )
        results["status"] = "ok"
    else:
        results["output_comparison"] = None
        results["status"] = "partial"
        results["failed_profiles"] = failures
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark.json"
    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False) as temporary:
        json.dump(results, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(report_path)
    print(f"Benchmark complete; report: {report_path}")
    if failures:
        raise SystemExit(
            "Benchmark was incomplete for profile(s): "
            + ", ".join(failures)
            + f". Inspect {report_path} and the preceding SeedVR2 error."
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="verify the complete local installation")
    doctor_parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    doctor_parser.add_argument(
        "--all-profiles", action="store_true", help="verify both FP16 and FP8 models"
    )
    run_parser = subparsers.add_parser("run", help="restore one image or an image directory")
    run_input = run_parser.add_mutually_exclusive_group(required=True)
    run_input.add_argument("--input-dir", type=Path, help="source image directory")
    run_input.add_argument("--input-image", type=Path, help="single source image")
    run_parser.add_argument(
        "--simple",
        action="store_true",
        help="restore one image in the current directory as NAME_restored.EXT",
    )
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument(
        "--work-dir",
        type=Path,
        help="intermediate directory (default: hidden sibling of OUTPUT_DIR)",
    )
    run_parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    run_parser.add_argument(
        "--cuda-blocks-to-swap",
        type=int,
        help="override the selected profile's tested-starting-point BlockSwap value (0-32)",
    )
    run_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include nested images (default: enabled; use --no-recursive to disable)",
    )
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace planned output files that already exist",
    )
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--resolution-quantum",
        type=int,
        default=256,
        help="internal model bucket quantum; saved dimensions remain unchanged",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="compare FP16 and FP8 on one copied image using CUDA"
    )
    benchmark_parser.add_argument("--input-image", type=Path, required=True)
    benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    benchmark_parser.add_argument("--work-dir", type=Path, required=True)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument(
        "--resolution-quantum",
        type=int,
        default=256,
        help="round the native short edge upward; must match production settings",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "doctor":
        doctor(list(PROFILES) if args.all_profiles else [args.profile])
    elif args.command == "benchmark":
        benchmark(args)
    elif args.simple:
        simple(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

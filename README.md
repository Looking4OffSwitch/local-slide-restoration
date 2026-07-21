# Local Slide Restoration

A local, automated pipeline for restoring scanned photographic slides on Apple Silicon Macs. It applies the same deterministic pipeline to every image, regardless of whether the source was captured with a camera and lightbox or digitized with a slide scanner.

The pipeline performs:

1. EXIF-aware orientation correction.
2. Conservative, image-adaptive color-cast correction in linear RGB.
3. Local contrast recovery in LAB color space.
4. SeedVR2 restoration using the Mac's Metal Performance Shaders (MPS) GPU backend.
5. LAB perceptual color matching to preserve the source image's identity.
6. Adaptive, edge-masked sharpening.
7. Lossless PNG master and high-quality JPEG delivery output.
8. A JSON manifest containing dimensions, settings, quality metrics, and SHA-256 hashes.

The current release targets macOS on Apple Silicon. It was developed and verified on a MacBook Pro with an M2 Max and 32 GB of unified memory.

## Safety: originals are never output targets

This project is designed around a strict non-destructive workflow:

- Never point the output or working directories at your source-photo directory.
- Copy photographs to a separate input directory before running the pipeline.
- The program reads copied inputs and writes prepared files, model output, masters, and delivery files to separate directories.
- It refuses input, output, or work paths inside repository-local `manual/` and `machine/` source directories.
- It refuses symlinked inputs that resolve back into either protected source directory.
- It refuses an output or work directory nested beneath the input directory.
- Source photographs, model files, virtual environments, working files, and outputs are excluded by `.gitignore`.

Keep an independent backup of irreplaceable photographs. No automated restoration process should be the only custodian of archival material.

## Requirements

- Apple Silicon Mac (`arm64`)
- macOS
- Git
- Internet access during installation
- Approximately 10 GB of free space for the Python environment, SeedVR2 checkout, and model weights, plus working/output space
- 32 GB unified memory recommended for full-resolution scans

The installer uses [uv](https://docs.astral.sh/uv/) exclusively to install Python 3.12, create the isolated virtual environment, and synchronize the locked dependencies. It does not invoke `pip` or `python -m venv`.

## Installation

Clone the repository and run the idempotent installer:

```bash
git clone https://github.com/Looking4OffSwitch/local-slide-restoration.git
cd local-slide-restoration
./setup_pipeline.sh
```

The installer:

- installs `uv` in the current user account if it is unavailable;
- installs Python 3.12 through `uv`;
- creates the virtual environment at `.slide_pipeline/venv` through `uv sync`;
- checks out SeedVR2 at the pinned commit `4490bd1f482e026674543386bb2a4d176da245b9`;
- downloads the 3B FP16 SeedVR2 model and FP16 VAE;
- verifies both model files using pinned SHA-256 hashes; and
- runs the installation doctor.

Re-running `./setup_pipeline.sh` is safe. Existing valid downloads and dependencies are reused, while hashes and the complete installation are checked again.

## Verify the installation

```bash
./run_pipeline.sh doctor
```

A healthy Apple Silicon installation reports JSON similar to:

```json
{
  "status": "ok",
  "python": "3.12.9",
  "torch": "2.13.0",
  "mps_available": true,
  "seedvr2_commit": "4490bd1f482e026674543386bb2a4d176da245b9",
  "models": [
    "seedvr2_ema_3b_fp16.safetensors",
    "ema_vae_fp16.safetensors"
  ]
}
```

## Prepare copied inputs

Create a directory that is separate from your originals, then copy images into it. For example:

```bash
mkdir -p "$HOME/slide-restoration-job/input"
cp -p /path/to/originals/*.JPG "$HOME/slide-restoration-job/input/"
```

Confirm that the copied directory contains the expected files before continuing. Do not move or delete the originals.

Supported image extensions are JPEG, PNG, TIFF, and WebP. Video files are not processed.

## Run the pipeline

```bash
./run_pipeline.sh \
  --input-dir "$HOME/slide-restoration-job/input" \
  --output-dir "$HOME/slide-restoration-job/output" \
  --work-dir "$HOME/slide-restoration-job/work"
```

For copied images in nested subdirectories, add `--recursive`:

```bash
./run_pipeline.sh \
  --input-dir "$HOME/slide-restoration-job/input" \
  --output-dir "$HOME/slide-restoration-job/output" \
  --work-dir "$HOME/slide-restoration-job/work" \
  --recursive
```

The output structure is:

```text
output/
├── delivery/       High-quality JPEG files, quality 95 with 4:4:4 chroma
├── masters/        Lossless PNG archival masters
└── manifest.json   Processing record, measurements, and hashes
```

The work directory contains prepared color-corrected images and raw SeedVR2 output. It can consume substantial disk space and may be removed manually after the final output has been reviewed and backed up.

## Command options

```text
--input-dir PATH           Directory containing copied input images (required)
--output-dir PATH          Separate directory for final output (required)
--work-dir PATH            Separate directory for intermediate files (required)
--recursive                Include supported images in subdirectories
--limit N                  Process only the first N sorted images
--seed N                   Deterministic SeedVR2 seed; default: 42
--resolution-quantum N     Round the native short edge upward; default: 256
```

The resolution bucket is never lower than the image's native short edge. For example, a 3312×2208 scan is processed at a 2304-pixel short edge and produces a 3456×2304 result. This avoids silently shrinking archival scans.

## Performance

A verified 3312×2208 scanner image took approximately 299 seconds (about five minutes) on an M2 Max with 32 GB unified memory and produced a 3456×2304 result. Runtime varies with resolution, orientation, available unified memory, thermal state, and other GPU activity.

Processing hundreds of full-resolution images on Apple Silicon will take many hours. The pipeline is deterministic and intended for unattended batches, but the first small batch should always be reviewed before committing to a complete archive.

## Quality expectations

The settings are deliberately conservative. The goal is to improve clarity, pixel structure, color balance, and local contrast while minimizing invented facial or object detail.

Automated restoration has limits:

- Small dust and scratch marks may remain.
- Severe emulsion damage can require manual inpainting.
- Strongly faded film stocks may need per-collection color tuning.
- A generative restoration model can occasionally invent plausible detail.

Review every final image at full resolution before treating it as an archival master. Preserve the physical slides and untouched digital captures independently.

## Reproducibility and third-party components

- Python dependencies are declared in `pyproject.toml` and pinned transitively in `uv.lock`.
- SeedVR2 integration is fetched from [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) at a fixed commit.
- Model downloads are validated against SHA-256 values in `setup_pipeline.sh`.
- The processing manifest records input hashes, output hashes, dimensions, timestamps, color-correction measurements, and focus scores.

SeedVR2 and its model weights are downloaded at installation time and are not distributed by this repository. Their respective licenses and usage terms apply separately.

## Troubleshooting

### `Pipeline is not installed`

Run:

```bash
./setup_pipeline.sh
```

### `Doctor failed: PyTorch MPS is unavailable`

Confirm that the machine is an Apple Silicon Mac and that the command is not running inside an environment that hides the Metal GPU. Then rerun the installer and doctor.

### Out-of-memory termination

Close GPU-intensive applications and retry the same settings. Do not lower resolution without consciously accepting the archival-quality tradeoff.

### Pipeline refuses a path

The path conflicts with a non-destructive safety rule. Create a new copied-input directory and separate output/work directories rather than bypassing the guard.

## License

The original code in this repository is available under the MIT License. Third-party repositories, Python packages, and model weights retain their own licenses.

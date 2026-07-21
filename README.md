# Local Slide Restoration

A guarded, local pipeline for restoring scanned photographic slides on:

- Apple Silicon macOS using PyTorch MPS; and
- x86-64 Bazzite Linux using an NVIDIA CUDA GPU.

The macOS path was verified on an M2 Max with 32 GB unified memory. The Bazzite path is an initial implementation for a system with an RTX 4080-class GPU, 12 GB VRAM, 64 GB system RAM, and the current Bazzite-provided NVIDIA driver. It must be benchmarked on that PC before production use.

The pipeline applies:

1. EXIF-aware orientation correction.
2. Conservative, image-adaptive color-cast correction in linear RGB.
3. Local contrast recovery in LAB color space.
4. SeedVR2 restoration on MPS or CUDA.
5. LAB perceptual color matching to preserve the source image's identity.
6. Adaptive, edge-masked sharpening.
7. Lossless PNG master and high-quality JPEG delivery output.
8. A JSON manifest with settings, dimensions, quality measurements, and SHA-256 hashes.

## Safety: originals are never output targets

- Work only with copied photographs. Keep the originals and an independent backup.
- Never point the output or working directories at the source-photo directory.
- The pipeline refuses the repository-local `originals/` archive and the legacy
  `manual/` and `machine/` source directories.
- It refuses symlinked inputs that resolve into either protected directory.
- It refuses an output or work directory nested beneath the input directory.
- Model files, environments, working files, and image outputs are excluded by `.gitignore`.

## Restoration profiles

Both profiles use the pinned 3B SeedVR2 architecture, the same FP16 VAE, the same seed, LAB color matching, tiled VAE encoding/decoding, and native-resolution bucketing.

| Profile | DiT model | CUDA starting configuration | Purpose |
|---|---|---|---|
| `archival-fp16` | 3B FP16 | 32 BlockSwap blocks, CPU offload | Highest available model precision; existing Mac default |
| `balanced-fp8` | 3B FP8 E4M3FN | 16 BlockSwap blocks, CPU offload | SeedVR2's recommended model class for 12–16 GB VRAM |

The CUDA BlockSwap values are explicit starting configurations, not claims of optimal tuning. The benchmark records them. Do not interpret the FP8/FP16 comparison metrics as a quality score; they measure how different the two outputs are. Visual review determines whether either output is acceptable.

SeedVR2's current hardware guidance is available in the [upstream project documentation](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler#requirements).

## Bazzite requirements

- Current x86-64 Bazzite NVIDIA image
- Operational Bazzite-provided NVIDIA driver and `nvidia-smi`
- NVIDIA CUDA-capable GPU; the initial target has 12 GB VRAM
- 64 GB system RAM for model and tensor offloading on the initial target
- Git, curl, and internet access during installation
- At least 20 GB free for the CUDA environment, pinned SeedVR2 checkout, both DiT models, and the VAE, plus separate working/output space

Bazzite supplies and updates NVIDIA drivers as part of its NVIDIA OS images. Do not install or replace the host driver for this pipeline. See the [Bazzite NVIDIA FAQ](https://docs.bazzite.gg/General/FAQ/#are-nvidia-graphics-card-drivers-pre-installed).

The installer uses `uv` in user-writable storage and does not modify Bazzite's read-only system image. PyTorch's locked Linux wheels provide the matching user-space CUDA libraries; the host provides the NVIDIA driver.

## Install on Bazzite

Clone the repository on the Bazzite PC and run:

```bash
git clone https://github.com/Looking4OffSwitch/local-slide-restoration.git
cd local-slide-restoration
./setup_bazzite.sh
```

The idempotent installer:

- verifies Bazzite, x86-64, `nvidia-smi`, and required host commands;
- reports the detected GPU, VRAM, and driver;
- installs Python 3.12 and the locked CUDA-capable environment through `uv`;
- checks out SeedVR2 at commit `4490bd1f482e026674543386bb2a4d176da245b9`;
- downloads the 3B FP16 model, 3B FP8 model, and FP16 VAE;
- verifies all three pinned SHA-256 hashes; and
- runs the complete CUDA doctor for both profiles.

Re-running `./setup_bazzite.sh` reuses valid downloads and checks the complete installation again.

Verify it independently with:

```bash
./run.sh doctor --all-profiles
```

The report must identify `Bazzite`, backend `cuda`, the expected NVIDIA GPU, approximately 12 GB VRAM, 64 GB system RAM, both profiles, the pinned SeedVR2 commit, and the installed NVIDIA driver. If any value is wrong, stop before benchmarking.

## Run the Bazzite FP16-versus-FP8 benchmark

First make a copied input. This example uses the scanner image included in the working project, but the path on the Bazzite PC may differ:

```bash
mkdir -p "$HOME/slide-restoration-benchmark/input"
cp -p /path/to/PICT0243.JPG "$HOME/slide-restoration-benchmark/input/"
```

Then run both profiles with identical image, seed, and resolution rules:

```bash
./run.sh benchmark \
  --input-image "$HOME/slide-restoration-benchmark/input/PICT0243.JPG" \
  --output-dir "$HOME/slide-restoration-benchmark/output" \
  --work-dir "$HOME/slide-restoration-benchmark/work"
```

For the included 3312×2208 scan, the default resolution quantum requests a 3456×2304 result from both profiles. The benchmark writes:

```text
output/
├── archival-fp16/
│   ├── delivery/
│   ├── masters/
│   └── manifest.json
├── balanced-fp8/
│   ├── delivery/
│   ├── masters/
│   └── manifest.json
└── benchmark.json
```

`benchmark.json` records end-to-end time, model name, BlockSwap value, dimensions, hashes, mean absolute channel difference, PSNR, and SSIM. PSNR and SSIM compare the two generated outputs to each other; they do not establish that one is more faithful to the physical slide.

If either SeedVR2 run fails, the benchmark still attempts the other profile and writes a `partial` report containing the failed command and return code. It then exits unsuccessfully so a missing comparison cannot be mistaken for a completed benchmark.

Review both PNG masters at full resolution. Do not choose a production profile from runtime alone.

## Run a production batch

Create a copied-input directory. The quality-first FP16 profile and recursive discovery
are the defaults, and the work directory defaults to a hidden sibling named
`.output-work`:

```bash
./run.sh \
  --input-dir "$HOME/slide-restoration-job/input" \
  --output-dir "$HOME/slide-restoration-job/output"
```

For the FP8 profile, replace `archival-fp16` with `balanced-fp8`.

The FP8 CUDA starting point uses 16 swapped blocks. If the Bazzite test produces a CUDA out-of-memory error, do not reduce archival resolution or change models silently. Retry deliberately with 24 and then 32 blocks:

```bash
./run.sh \
  --profile balanced-fp8 \
  --cuda-blocks-to-swap 24 \
  --input-dir "$HOME/slide-restoration-job/input" \
  --output-dir "$HOME/slide-restoration-job/output-fp8-24" \
  --work-dir "$HOME/slide-restoration-job/work-fp8-24"
```

The FP16 profile already starts at the maximum 32 blocks. If it still fails, preserve the error and hardware report for diagnosis rather than falling back to FP8 automatically.

## Install and run on Apple Silicon

The existing Mac workflow remains FP16 by default:

```bash
./setup_pipeline.sh
./run.sh doctor
./run.sh \
  --input-dir "$HOME/slide-restoration-job/input" \
  --output-dir "$HOME/slide-restoration-job/output"
```

Requirements are Apple Silicon, macOS, Git, internet access during installation, and approximately 10 GB for the FP16 installation plus working/output space. A 32 GB unified-memory Mac is recommended for full-resolution scans.

## Common command options

```text
--input-dir PATH              Copied input directory (required for run)
--output-dir PATH             Separate final-output directory (required)
--work-dir PATH               Intermediate directory; defaults to .OUTPUT-work beside output
--profile NAME                archival-fp16 or balanced-fp8
--cuda-blocks-to-swap N       Explicit CUDA override, 0-32
--recursive / --no-recursive  Include nested images; enabled by default
--limit N                     Process the first N sorted images
--seed N                      Deterministic SeedVR2 seed; default 42
--resolution-quantum N        Native short-edge bucket; default 256
```

Supported inputs are JPEG, PNG, TIFF, and WebP. Video files are not processed.

The resolution bucket is never lower than the native short edge. This avoids silently shrinking archival scans.

## Output and reproducibility

```text
output/
├── delivery/       JPEG quality 95 with 4:4:4 chroma
├── masters/        Lossless PNG archival masters
└── manifest.json   Processing settings, measurements, and hashes
```

The work directory contains prepared color-corrected images and raw SeedVR2 output. It can be large and may be removed manually only after the final results have been reviewed and backed up.

Dependencies are declared in `pyproject.toml` and pinned transitively in `uv.lock`. SeedVR2 is pinned to a specific Git commit. Model downloads are verified against pinned hashes. The manifest records the backend, selected profile, model, CUDA BlockSwap value, input/output hashes, dimensions, timestamps, color measurements, and focus score.

## Quality limits

The settings are conservative, but SeedVR2 is generative:

- Dust and scratches may remain.
- Severe emulsion damage may require manual inpainting.
- Strongly faded film stocks may need collection-specific color tuning.
- Plausible but invented facial or object detail is possible.
- FP8 uses lower-precision model weights and can differ from FP16.

Review every master at full resolution. Preserve the physical slides and untouched digital captures independently.

## Troubleshooting

### `Pipeline is not installed`

Run `./setup_bazzite.sh` on Bazzite or `./setup_pipeline.sh` on Apple Silicon.

### Bazzite doctor reports that CUDA is unavailable

Confirm that `nvidia-smi` succeeds and that the PC is using the current Bazzite NVIDIA image. Do not install a separate NVIDIA driver over the Bazzite-managed driver.

### CUDA out of memory

Keep the copied input, resolution, seed, and profile fixed. For FP8, increase BlockSwap from 16 to 24 and then 32. FP16 already uses 32. Do not lower resolution unless consciously accepting a different archival result.

### Pipeline refuses a path

The path conflicts with a non-destructive safety rule. Create a new copied-input directory and separate output/work directories instead of bypassing the guard.

## License

Original code in this repository is available under the MIT License. SeedVR2, model weights, and other third-party components retain their own licenses and usage terms.

# Local Slide Restoration

The production path is a fidelity-first local workflow for photographed slides. It uses
only source-derived image operations: conservative edge-mounted border trimming,
anti-aliased demoiré/downsampling, restrained cast and contrast correction, and mild
sharpening. It does not run a generative model or face enhancer, so it cannot invent
faces, text, hands, clothing, or background objects.

The previous Qwen-Image-Edit-2511 workflow remains available as an explicitly generative
experimental path. It is not the production default because whole-frame diffusion edits
can replace uncertain historical detail with plausible but unsupported content.

## Install on the Bazzite RTX 4080 PC

On the PC:

```bash
git clone https://github.com/Looking4OffSwitch/local-slide-restoration.git
cd local-slide-restoration
./setup_bazzite.sh
```

The installer is idempotent and uses user-writable repository storage. It:

- verifies x86-64 Bazzite, the NVIDIA driver, RTX 4080, approximately 12 GB VRAM,
  at least 60 GiB usable system RAM, and at least 50 GiB free for the first installation;
- installs pinned ComfyUI and ComfyUI-GGUF checkouts with a private Python environment;
- downloads the shared Qwen encoder, VAE, and Lightning adapter;
- downloads both Q4_K_S and Q4_K_M diffusion models and verifies every model SHA-256;
- confirms that PyTorch in ComfyUI can actually access CUDA; and
- runs the repository workflow/model doctor for both test profiles.

Bazzite owns the host NVIDIA driver. The installer does not replace or modify it.
It installs the stable NVIDIA PyTorch package from the CUDA 13.0 index specified by the
[ComfyUI system requirements](https://docs.comfy.org/installation/system_requirements).

## Restore one image

Run the simple command from the directory where the restored image should be written:

```bash
./run.sh --simple --input-image originals/manual/IMG_6219.jpeg
```

This uses the non-generative fidelity engine and writes
`IMG_6219_restored.jpeg` to the caller's current directory. The output keeps the source
extension and contains image data in that actual format. Existing output is refused
unless `--overwrite` is supplied. Nothing is written inside the protected `originals/`
archive.

The default maximum output dimension is 1200 pixels. The primary demoiré stage uses the
pinned Apache-2.0 ESDNet UHDM model, which was trained specifically for camera-captured
4K screens and does not use a semantic generative prior. If it is unavailable, the runner
falls back to source-derived low-pass filtering. Override the output size when needed:

```bash
./run.sh --simple --input-image originals/manual/IMG_6219.jpeg --max-dimension 2400
```

Use `--demoire esdnet` to require the model and fail instead of falling back, or
`--demoire filter` to force the deterministic fallback.

To deliberately run the previous Qwen workflow for comparison:

```bash
./run.sh --simple --generative --input-image originals/manual/IMG_6219.jpeg
```

## Compare both model profiles

To make a deliberate quality comparison between both installed model profiles:

```bash
./run.sh comfy-benchmark \
  --input-image /path/to/original.jpeg \
  --output-dir "$HOME/slide-restoration-test"
```

This creates:

```text
slide-restoration-test/
├── original_q4ks_restored.png
├── original_q4km_restored.png
└── benchmark.json
```

`q4ks` is the smaller 12 GB-PC candidate. `q4km` is the Mac-tested quality reference
that produced the strong restoration during development, but it may require more CUDA
offloading. The PC comparison deliberately runs both rather than assuming either is the
right production choice. `benchmark.json` records each model, output hash, output size,
generation time, and total time. If a profile fails, the other is still attempted and
the command exits unsuccessfully with the failure recorded.

Review both images at full size for identity, composition, facial detail, invented
content, mount removal, moire removal, exposure, and color. Do not start the 600-image
batch until one profile passes that review and its measured PC time is acceptable.

Re-run the complete installation check at any time with:

```bash
./run.sh comfy-doctor --all-profiles --require-bazzite-cuda
```

## Advanced single-image usage

The explicit command remains available when a PNG output path or Q4_K_M is required:

```bash
./run.sh comfy \
  --profile q4ks \
  --input-image /path/to/copied-slide.jpeg \
  --output-image /path/to/copied-slide_restored.png
```

Use `--profile q4km` only if that is the result selected from the PC comparison.

## Restore the 600-image batch

Keep source copies and output in separate directories, then run one persistent ComfyUI
process:

```bash
./run.sh comfy-batch \
  --input-dir /path/to/copied-inputs \
  --output-dir /path/to/restored-output
```

This uses the non-generative fidelity engine. Discovery is recursive, relative
subdirectories are preserved, and outputs are named
`NAME_restored.png`. Existing results are skipped by default, so the same command resumes
an interrupted batch. `restoration_manifest.json` records completed, skipped, and failed
sources with per-image timings. A failed image does not stop the rest unless `--fail-fast`
is supplied.

The experimental generative batch remains available with `--generative --profile q4ks`
or `--generative --profile q4km`. Generated outputs must be reviewed individually and
must not replace the archival fidelity results.

Use the benchmark's measured generation time—not a Mac estimate—to project any
experimental Qwen batch runtime on the RTX PC.

## Inspect the ComfyUI graph

```bash
./run.sh comfy-ui --profile q4ks
```

Open `http://127.0.0.1:8188` and load
`workflows/photo_restoration_qwen_2511.json`. The loadable graph mirrors the automated
API graph in `workflows/photo_restoration_qwen_2511_api.json`. To inspect Q4_K_M, select
`qwen-image-edit-2511-Q4_K_M.gguf` in the diffusion-model node.

The restoration instructions live in
`workflows/photo_restoration_prompt.txt`. After editing them, regenerate the loadable
canvas graph with:

```bash
python3 tools/build_comfy_workflow.py
```

## macOS compatibility

The same graph and runner remain usable on Apple Silicon. A local Mac installation can
be created with:

```bash
./setup_comfyui.sh --install-comfyui --model-set q4km
./run.sh comfy-doctor --profile q4km
```

This compatibility path does not change the production target: profile selection and
performance acceptance are based on the Bazzite RTX 4080 test.

## Safety

- The repository's `originals/` directory is permanently read-only source material.
- Generated images, work files, models, manifests, and caches are never written there.
- Single-image output cannot replace its source.
- Batch output must be outside the batch input directory.
- Models, environments, source photographs, generated images, and archives are ignored
  by Git.

## License

Original code in this repository is available under the MIT License. ComfyUI, custom
nodes, and model weights retain their own licenses and usage terms.

The optional ESDNet checkout is installed from `CVMI-Lab/UHDM` at a pinned commit and is
licensed under Apache-2.0. Its checkpoint is stored outside Git under `.slide_pipeline`.

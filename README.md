# Local Slide Restoration

The production path is a local ComfyUI workflow using Qwen-Image-Edit-2511 and
the official four-step Lightning adapter. It is designed for photographed slides that
need the mount removed, geometry corrected, capture artifacts cleaned, and the original
scene restored conservatively.

The workflow does not run a post-restoration upscaler. Qwen normalizes the source to its
native working resolution, performs the edit, and saves that result directly. The older
SeedVR2 enhancement pipeline is still present for reproducibility, but it is not the
production restoration solution.

## Test on the Bazzite RTX 4080 PC

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

Copy the supplied `original.jpeg` to the PC, then run the real comparison:

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

## Restore one image after choosing a profile

Profile selection is required so the software never silently chooses quality versus
memory use:

```bash
./run.sh comfy \
  --profile q4ks \
  --input-image /path/to/copied-slide.jpeg \
  --output-image /path/to/copied-slide_restored.png
```

Use `--profile q4km` only if that is the result selected from the PC comparison. Existing
outputs are refused unless `--overwrite` is supplied.

## Restore the 600-image batch

Keep source copies and output in separate directories, then run one persistent ComfyUI
process:

```bash
./run.sh comfy-batch \
  --profile q4ks \
  --input-dir /path/to/copied-inputs \
  --output-dir /path/to/restored-output
```

Replace `q4ks` with `q4km` if that profile wins the PC comparison. The model remains
resident for the whole batch instead of being reloaded for every photograph. Discovery
is recursive, relative subdirectories are preserved, and outputs are named
`NAME_restored.png`. Existing results are skipped by default, so the same command resumes
an interrupted batch. `restoration_manifest.json` records completed, skipped, and failed
sources with per-image timings. A failed image does not stop the rest unless `--fail-fast`
is supplied.

Use the benchmark's measured generation time—not a Mac estimate—to project the 600-image
runtime on the RTX PC.

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

## Legacy SeedVR2 path

The previous `slide_pipeline.py`, `setup_pipeline.sh`, and legacy `./run.sh` commands
remain available for reproducing earlier enhancement tests. They are not used by any
`comfy-*` command and are not the recommended solution for this project.

## License

Original code in this repository is available under the MIT License. ComfyUI, custom
nodes, and model weights retain their own licenses and usage terms.

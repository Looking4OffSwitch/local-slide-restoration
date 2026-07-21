#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PIPELINE_DIR="$SCRIPT_DIR/.slide_pipeline"
VENDOR_DIR="$PIPELINE_DIR/vendor/seedvr2"
VENV_DIR="$PIPELINE_DIR/venv"
MODEL_DIR="$PIPELINE_DIR/models"
DOWNLOAD_DIR="$PIPELINE_DIR/downloads"
SEEDVR2_REPO="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_COMMIT="4490bd1f482e026674543386bb2a4d176da245b9"
FP16_DIT_MODEL="seedvr2_ema_3b_fp16.safetensors"
FP8_DIT_MODEL="seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE_MODEL="ema_vae_fp16.safetensors"
FP16_DIT_SHA256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"
FP8_DIT_SHA256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"
VAE_SHA256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"

log() {
  printf '[slide-pipeline] %s\n' "$*"
}

fail() {
  printf '[slide-pipeline] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command is unavailable: $1"
}

[[ "$(uname -s)" == "Linux" ]] || fail "This installer targets Bazzite Linux."
[[ "$(uname -m)" == "x86_64" ]] || fail "This installer requires x86_64 Bazzite."
[[ -r /etc/os-release ]] || fail "Cannot read /etc/os-release."
OS_ID="$(awk -F= '$1 == "ID" {gsub(/"/, "", $2); print $2}' /etc/os-release)"
[[ "$OS_ID" == "bazzite" ]] || fail "This installer requires Bazzite (reported ID=$OS_ID)."
[[ -f "$SCRIPT_DIR/slide_pipeline.py" ]] || \
  fail "Missing slide_pipeline.py next to this installer."

require_command curl
require_command git
require_command nvidia-smi
require_command sha256sum

GPU_REPORT="$(nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader,nounits)" || fail "The NVIDIA driver is not operational."
log "Detected NVIDIA hardware"
printf '%s\n' "$GPU_REPORT"
GPU_FIRST_LINE="$(printf '%s\n' "$GPU_REPORT" | sed -n '1p')"
GPU_NAME="$(printf '%s\n' "$GPU_FIRST_LINE" | awk -F, '{gsub(/^ +| +$/, "", $1); print $1}')"
GPU_MEMORY_MIB="$(printf '%s\n' "$GPU_FIRST_LINE" | awk -F, '{gsub(/^ +| +$/, "", $2); print $2}')"
[[ "$GPU_NAME" == *"RTX 4080"* ]] || fail "GPU 0 is not the expected RTX 4080: $GPU_NAME"
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || fail "Could not parse GPU 0 VRAM: $GPU_MEMORY_MIB"
(( GPU_MEMORY_MIB >= 11264 && GPU_MEMORY_MIB <= 13312 )) || \
  fail "GPU 0 does not report the expected 12 GB VRAM: ${GPU_MEMORY_MIB} MiB"
SYSTEM_RAM_KIB="$(awk '$1 == "MemTotal:" {print $2}' /proc/meminfo)"
[[ "$SYSTEM_RAM_KIB" =~ ^[0-9]+$ ]] || fail "Could not read system RAM from /proc/meminfo."
(( SYSTEM_RAM_KIB >= 62914560 )) || fail "Host reports less than the expected 64 GB system RAM."

mkdir -p "$PIPELINE_DIR" "$PIPELINE_DIR/vendor" "$MODEL_DIR" "$DOWNLOAD_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  log "Installing uv into the current user account"
  UV_INSTALLER="$DOWNLOAD_DIR/uv-install.sh"
  curl --fail --location --retry 3 --output "$UV_INSTALLER" \
    "https://astral.sh/uv/install.sh"
  sh "$UV_INSTALLER"
  UV_BIN="$HOME/.local/bin/uv"
fi
[[ -x "$UV_BIN" ]] || fail "uv is unavailable after installation."

if [[ -d "$VENDOR_DIR/.git" ]]; then
  if ! git -C "$VENDOR_DIR" diff --quiet || ! git -C "$VENDOR_DIR" diff --cached --quiet; then
    fail "The pinned SeedVR2 checkout has local changes: $VENDOR_DIR"
  fi
  log "Refreshing the pinned SeedVR2 checkout"
  git -C "$VENDOR_DIR" fetch --depth 1 origin "$SEEDVR2_COMMIT"
else
  [[ ! -e "$VENDOR_DIR" ]] || fail "$VENDOR_DIR exists but is not a Git checkout."
  log "Downloading SeedVR2"
  git clone --no-checkout --filter=blob:none "$SEEDVR2_REPO" "$VENDOR_DIR"
  git -C "$VENDOR_DIR" fetch --depth 1 origin "$SEEDVR2_COMMIT"
fi
git -C "$VENDOR_DIR" checkout --detach "$SEEDVR2_COMMIT"

log "Creating the Python 3.12 CUDA environment in user-writable storage"
"$UV_BIN" python install 3.12 --no-bin
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync \
  --project "$SCRIPT_DIR" \
  --python 3.12 \
  --locked \
  --no-dev

log "Downloading and validating both SeedVR2 comparison profiles"
(
  cd "$VENDOR_DIR"
  "$VENV_DIR/bin/python" - \
    "$MODEL_DIR" "$FP16_DIT_MODEL" "$FP8_DIT_MODEL" "$VAE_MODEL" <<'PY'
import sys
from src.utils.downloads import download_weight

model_dir, fp16_model, fp8_model, vae_model = sys.argv[1:]
for dit_model in (fp16_model, fp8_model):
    if not download_weight(dit_model, vae_model, model_dir=model_dir):
        raise SystemExit(f"model download or validation failed: {dit_model}")
PY
)

verify_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "SHA-256 mismatch for $path"
}
verify_hash "$FP16_DIT_SHA256" "$MODEL_DIR/$FP16_DIT_MODEL"
verify_hash "$FP8_DIT_SHA256" "$MODEL_DIR/$FP8_DIT_MODEL"
verify_hash "$VAE_SHA256" "$MODEL_DIR/$VAE_MODEL"

log "Running the complete Bazzite/CUDA installation self-check"
SLIDE_PIPELINE_ROOT="$SCRIPT_DIR" \
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/slide_pipeline.py" doctor --all-profiles

log "Bazzite installation complete"
log "Next: $SCRIPT_DIR/run.sh benchmark --help"

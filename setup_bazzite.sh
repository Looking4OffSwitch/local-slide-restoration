#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
STATE_DIR="$SCRIPT_DIR/.slide_pipeline"
UV_DIR="$STATE_DIR/uv-bin"
UV_REQUIRED_VERSION="0.11.30"

log() {
  printf '[comfy-bazzite] %s\n' "$*"
}

fail() {
  printf '[comfy-bazzite] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command is unavailable: $1"
}

[[ "$(uname -s)" == "Linux" ]] || fail "This installer targets Bazzite Linux."
[[ "$(uname -m)" == "x86_64" ]] || fail "This installer requires x86-64 Bazzite."
[[ -r /etc/os-release ]] || fail "Cannot read /etc/os-release."
OS_ID="$(awk -F= '$1 == "ID" {gsub(/"/, "", $2); print $2}' /etc/os-release)"
[[ "$OS_ID" == "bazzite" ]] || fail "Expected Bazzite; /etc/os-release reports ID=$OS_ID."

require_command curl
require_command git
require_command nvidia-smi
require_command python3

mkdir -p "$STATE_DIR" "$UV_DIR"
if [[ -x "$UV_DIR/uv" ]]; then
  UV_BIN="$UV_DIR/uv"
else
  log "Installing uv $UV_REQUIRED_VERSION in repository-local storage"
  UV_INSTALLER="$STATE_DIR/uv-install.sh"
  curl --fail --location --retry 3 --output "$UV_INSTALLER" \
    "https://astral.sh/uv/$UV_REQUIRED_VERSION/install.sh"
  UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 \
    sh "$UV_INSTALLER"
  UV_BIN="$UV_DIR/uv"
fi
[[ -x "$UV_BIN" ]] || fail "uv is unavailable after installation."
[[ "$($UV_BIN --version)" == "uv $UV_REQUIRED_VERSION "* ]] || \
  fail "uv $UV_REQUIRED_VERSION is required; found $($UV_BIN --version)."

GPU_REPORT="$(nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader,nounits)" || fail "The Bazzite NVIDIA driver is not operational."
GPU_FIRST_LINE="$(printf '%s\n' "$GPU_REPORT" | sed -n '1p')"
GPU_NAME="$(printf '%s\n' "$GPU_FIRST_LINE" | awk -F, '{gsub(/^ +| +$/, "", $1); print $1}')"
GPU_MEMORY_MIB="$(printf '%s\n' "$GPU_FIRST_LINE" | awk -F, '{gsub(/^ +| +$/, "", $2); print $2}')"
[[ "$GPU_NAME" == *"RTX 4080"* ]] || fail "Expected the RTX 4080 target; detected $GPU_NAME."
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || fail "Could not parse GPU VRAM: $GPU_MEMORY_MIB"
(( GPU_MEMORY_MIB >= 11264 && GPU_MEMORY_MIB <= 13312 )) || \
  fail "Expected approximately 12 GB VRAM; detected ${GPU_MEMORY_MIB} MiB."

SYSTEM_RAM_KIB="$(awk '$1 == "MemTotal:" {print $2}' /proc/meminfo)"
[[ "$SYSTEM_RAM_KIB" =~ ^[0-9]+$ ]] || fail "Could not read system RAM."
(( SYSTEM_RAM_KIB >= 62914560 )) || fail "The host reports less than 60 GiB usable RAM."

AVAILABLE_KIB="$(df -Pk "$SCRIPT_DIR" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || fail "Could not measure available disk space."
if [[ ! -x "$STATE_DIR/ComfyUI/.venv/bin/python" || \
      ! -f "$STATE_DIR/comfyui_models/diffusion_models/qwen-image-edit-2511-Q4_K_S.gguf" || \
      ! -f "$STATE_DIR/comfyui_models/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf" || \
      ! -f "$STATE_DIR/comfyui_models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" || \
      ! -f "$STATE_DIR/comfyui_models/vae/qwen_image_vae.safetensors" || \
      ! -f "$STATE_DIR/comfyui_models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" ]]; then
  (( AVAILABLE_KIB >= 52428800 )) || \
    fail "A first installation requires at least 50 GiB free for dependencies and both models."
fi

log "Detected $GPU_NAME with ${GPU_MEMORY_MIB} MiB VRAM"
printf '%s\n' "$GPU_REPORT"
log "Installing the pinned ComfyUI stack and both comparison models"
UV_CACHE_DIR="$STATE_DIR/uv-cache" \
UV_PYTHON_INSTALL_DIR="$STATE_DIR/uv-python" \
PATH="$UV_DIR:$PATH" \
  "$SCRIPT_DIR/setup_comfyui.sh" \
  --install-comfyui \
  --model-set comparison \
  --nvidia

log "Running the Bazzite, CUDA, workflow, node, and model self-check"
"$SCRIPT_DIR/run.sh" comfy-doctor --all-profiles --require-bazzite-cuda

log "Installation is complete."
log "Next: ./run.sh --simple --input-image /path/to/original.jpeg"

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
DIT_MODEL="seedvr2_ema_3b_fp16.safetensors"
VAE_MODEL="ema_vae_fp16.safetensors"
DIT_SHA256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"
VAE_SHA256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"

log() {
  printf '[slide-pipeline] %s\n' "$*"
}

fail() {
  printf '[slide-pipeline] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$(uname -s)" == "Darwin" ]] || fail "This installer version targets macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "This installer requires Apple Silicon."
[[ -f "$SCRIPT_DIR/slide_pipeline.py" ]] || fail "Missing slide_pipeline.py next to this installer."

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

log "Creating the Python 3.12 environment"
"$UV_BIN" python install 3.12
log "Creating and synchronizing the uv-managed virtual environment"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync \
  --project "$SCRIPT_DIR" \
  --python 3.12 \
  --locked \
  --no-dev

log "Downloading and validating SeedVR2 model weights"
(
  cd "$VENDOR_DIR"
  "$VENV_DIR/bin/python" - "$MODEL_DIR" "$DIT_MODEL" "$VAE_MODEL" <<'PY'
import sys
from src.utils.downloads import download_weight

model_dir, dit_model, vae_model = sys.argv[1:]
if not download_weight(dit_model, vae_model, model_dir=model_dir):
    raise SystemExit("model download or validation failed")
PY
)

verify_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "SHA-256 mismatch for $path"
}
verify_hash "$DIT_SHA256" "$MODEL_DIR/$DIT_MODEL"
verify_hash "$VAE_SHA256" "$MODEL_DIR/$VAE_MODEL"

log "Running installation self-check"
SLIDE_PIPELINE_ROOT="$SCRIPT_DIR" "$VENV_DIR/bin/python" "$SCRIPT_DIR/slide_pipeline.py" doctor

log "Installation complete"
log "Run: $SCRIPT_DIR/run_pipeline.sh --help"

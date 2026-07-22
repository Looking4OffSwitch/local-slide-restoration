#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
STATE_DIR="$SCRIPT_DIR/.slide_pipeline"
MODEL_DIR="$STATE_DIR/comfyui_models"
COMFYUI_COMMIT="8b099de36acd81acd1afa3b5442951dc847e0a52"
GGUF_COMMIT="6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
ESDNET_COMMIT="fa70a92d3d4f35d5c4e3fa7a54e3b2e5b995f1cd"
ESDNET_SHA256="254235cd25f90a3f1785885385dc6cb3f2178e053291ab53d1943bd7c2f7de65"

usage() {
  cat <<'EOF'
Usage: ./setup_comfyui.sh [--comfyui-dir PATH] [--install-comfyui] [--model-set NAME] [--nvidia]

Installs the pinned photo-restoration models and ComfyUI-GGUF node used by the
repository workflow. Existing ComfyUI installations are reused by default.

Options:
  --comfyui-dir PATH   Use this ComfyUI checkout.
  --install-comfyui    Install the pinned ComfyUI checkout under .slide_pipeline.
  --model-set NAME     q4ks, q4km, or comparison (both). Default: q4km.
  --nvidia             Install the official stable NVIDIA CUDA PyTorch build.
  -h, --help           Show this help.
EOF
}

requested_comfy="${COMFYUI_DIR:-}"
install_comfy=0
model_set="q4km"
nvidia=0
while (($#)); do
  case "$1" in
    --comfyui-dir)
      [[ $# -ge 2 ]] || { printf '%s\n' '--comfyui-dir requires a path' >&2; exit 2; }
      requested_comfy="$2"
      shift 2
      ;;
    --install-comfyui)
      install_comfy=1
      shift
      ;;
    --model-set)
      [[ $# -ge 2 ]] || { printf '%s\n' '--model-set requires a value' >&2; exit 2; }
      model_set="$2"
      shift 2
      ;;
    --nvidia)
      nvidia=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$model_set" in
  q4ks|q4km|comparison) ;;
  *) printf 'Invalid --model-set: %s\n' "$model_set" >&2; exit 2 ;;
esac

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Required command is missing: %s\n' "$1" >&2
    exit 1
  }
}

require_command curl
require_command git

if [[ -z "$requested_comfy" && $install_comfy -eq 0 ]]; then
  candidates=(
    "$HOME/ComfyUI"
    "$HOME/dev/ComfyUI"
    "$HOME/dev/3rd_party/ComfyUI-Easy-Install/ComfyUI-Easy-Install/ComfyUI"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/main.py" ]]; then
      requested_comfy="$candidate"
      break
    fi
  done
fi

if [[ $install_comfy -eq 1 ]]; then
  requested_comfy="$STATE_DIR/ComfyUI"
  if [[ ! -d "$requested_comfy/.git" ]]; then
    mkdir -p "$STATE_DIR"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$requested_comfy"
  fi
  if ! git -C "$requested_comfy" diff --quiet || \
     ! git -C "$requested_comfy" diff --cached --quiet; then
    printf 'Refusing to change a modified ComfyUI checkout: %s\n' "$requested_comfy" >&2
    exit 1
  fi
  git -C "$requested_comfy" fetch --depth 1 origin "$COMFYUI_COMMIT"
  git -C "$requested_comfy" checkout --detach "$COMFYUI_COMMIT"
fi

if [[ -z "$requested_comfy" || ! -f "$requested_comfy/main.py" ]]; then
  printf '%s\n' 'ComfyUI was not found.' >&2
  printf '%s\n' 'Pass --comfyui-dir PATH, set COMFYUI_DIR, or use --install-comfyui.' >&2
  exit 1
fi

COMFYUI_DIR_RESOLVED="$(cd "$requested_comfy" && pwd -P)"
comfy_python=""
for candidate in \
  "$COMFYUI_DIR_RESOLVED/../python_embeded/python" \
  "$COMFYUI_DIR_RESOLVED/../python_embeded/python.exe" \
  "$COMFYUI_DIR_RESOLVED/.venv/bin/python" \
  "$COMFYUI_DIR_RESOLVED/venv/bin/python" \
  "$COMFYUI_DIR_RESOLVED/venv/Scripts/python.exe"; do
  if [[ -x "$candidate" ]]; then
    comfy_python="$candidate"
    break
  fi
done

if [[ -z "$comfy_python" && $install_comfy -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12 --no-bin
    uv venv --python 3.12 "$COMFYUI_DIR_RESOLVED/.venv"
    comfy_python="$COMFYUI_DIR_RESOLVED/.venv/bin/python"
  else
    interpreters=(python3.13 python3.12 python3.11 python3)
    for interpreter in "${interpreters[@]}"; do
      if [[ -x "$interpreter" ]] || command -v "$interpreter" >/dev/null 2>&1; then
        if ! "$interpreter" -c \
          'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
          continue
        fi
        "$interpreter" -m venv "$COMFYUI_DIR_RESOLVED/.venv"
        comfy_python="$COMFYUI_DIR_RESOLVED/.venv/bin/python"
        break
      fi
    done
  fi
  [[ -n "$comfy_python" ]] || {
    printf '%s\n' 'Python 3.11, 3.12, or 3.13 is required.' >&2
    exit 1
  }
fi

if [[ -z "$comfy_python" ]]; then
  printf '%s\n' 'Could not find the Python environment used by this ComfyUI installation.' >&2
  printf '%s\n' 'Install ComfyUI dependencies first or rerun with --install-comfyui.' >&2
  exit 1
fi

if [[ $install_comfy -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    if [[ $nvidia -eq 1 ]]; then
      uv pip install --python "$comfy_python" \
        torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu130
    fi
    uv pip install --python "$comfy_python" -r "$COMFYUI_DIR_RESOLVED/requirements.txt"
  else
    "$comfy_python" -m pip install --upgrade pip
    if [[ $nvidia -eq 1 ]]; then
      "$comfy_python" -m pip install \
        torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu130
    fi
    "$comfy_python" -m pip install -r "$COMFYUI_DIR_RESOLVED/requirements.txt"
  fi
fi

gguf_dir="$COMFYUI_DIR_RESOLVED/custom_nodes/ComfyUI-GGUF"
if [[ ! -d "$gguf_dir/.git" ]]; then
  git clone https://github.com/city96/ComfyUI-GGUF.git "$gguf_dir"
fi
if [[ "$(git -C "$gguf_dir" rev-parse HEAD)" != "$GGUF_COMMIT" ]]; then
  if ! git -C "$gguf_dir" diff --quiet || ! git -C "$gguf_dir" diff --cached --quiet; then
    printf 'Refusing to change a modified ComfyUI-GGUF checkout: %s\n' "$gguf_dir" >&2
    exit 1
  fi
  git -C "$gguf_dir" fetch --depth 1 origin "$GGUF_COMMIT"
  git -C "$gguf_dir" checkout --detach "$GGUF_COMMIT"
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$comfy_python" -r "$gguf_dir/requirements.txt"
  uv pip install --python "$comfy_python" gdown
else
  "$comfy_python" -m pip install -r "$gguf_dir/requirements.txt"
  "$comfy_python" -m pip install gdown
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download_model() {
  local relative_path="$1"
  local url="$2"
  local expected_sha="$3"
  local destination="$MODEL_DIR/$relative_path"
  local partial="$destination.part"

  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" && "$(sha256_file "$destination")" == "$expected_sha" ]]; then
    printf 'Verified: %s\n' "$relative_path"
    return
  fi
  if [[ -f "$destination" ]]; then
    printf 'Checksum mismatch; replacing: %s\n' "$relative_path" >&2
    mv "$destination" "$destination.invalid.$(date +%s)"
  fi
  printf 'Downloading: %s\n' "$relative_path"
  curl --fail --location --continue-at - --output "$partial" "$url"
  local actual_sha
  actual_sha="$(sha256_file "$partial")"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    printf 'Checksum mismatch for %s\nexpected: %s\nactual:   %s\n' \
      "$relative_path" "$expected_sha" "$actual_sha" >&2
    exit 1
  fi
  mv "$partial" "$destination"
}

esdnet_dir="$STATE_DIR/third_party/ESDNet"
if [[ ! -d "$esdnet_dir/.git" ]]; then
  mkdir -p "$(dirname "$esdnet_dir")"
  git clone https://github.com/CVMI-Lab/UHDM.git "$esdnet_dir"
fi
if [[ "$(git -C "$esdnet_dir" rev-parse HEAD)" != "$ESDNET_COMMIT" ]]; then
  if ! git -C "$esdnet_dir" diff --quiet || ! git -C "$esdnet_dir" diff --cached --quiet; then
    printf 'Refusing to change a modified ESDNet checkout: %s\n' "$esdnet_dir" >&2
    exit 1
  fi
  git -C "$esdnet_dir" fetch --depth 1 origin "$ESDNET_COMMIT"
  git -C "$esdnet_dir" checkout --detach "$ESDNET_COMMIT"
fi
esdnet_model="$STATE_DIR/restoration_models/esdnet_uhdm.pth"
mkdir -p "$(dirname "$esdnet_model")"
if [[ ! -f "$esdnet_model" || "$(sha256_file "$esdnet_model")" != "$ESDNET_SHA256" ]]; then
  rm -f "$esdnet_model"
  "$comfy_python" -m gdown \
    'https://drive.google.com/uc?id=1HT_ubcAYRqzFIJ46XuPhrulJk2YFBIEl' \
    -O "$esdnet_model"
fi
if [[ "$(sha256_file "$esdnet_model")" != "$ESDNET_SHA256" ]]; then
  printf '%s\n' 'ESDNet checksum mismatch after download.' >&2
  exit 1
fi
printf 'Verified: %s\n' "restoration_models/esdnet_uhdm.pth"

if [[ "$model_set" == "q4ks" || "$model_set" == "comparison" ]]; then
  download_model \
    "diffusion_models/qwen-image-edit-2511-Q4_K_S.gguf" \
    "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_S.gguf" \
    "df952ef0d2b46463bd95d9afbb78e045ec5412316f453a7ad5a3d7bcbb111b72"
fi
if [[ "$model_set" == "q4km" || "$model_set" == "comparison" ]]; then
  download_model \
    "diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf" \
    "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf" \
    "8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723"
fi
download_model \
  "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"
download_model \
  "vae/qwen_image_vae.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
  "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
download_model \
  "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
  "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"
printf '\nComfyUI restoration installation is complete.\n'
printf 'ComfyUI: %s\n' "$COMFYUI_DIR_RESOLVED"
printf 'Python:  %s\n' "$comfy_python"
doctor_profiles=(--profile "$model_set")
if [[ "$model_set" == "comparison" ]]; then
  doctor_profiles=(--all-profiles)
fi
"$comfy_python" "$SCRIPT_DIR/comfy_restore.py" doctor \
  --comfyui-dir "$COMFYUI_DIR_RESOLVED" \
  --comfyui-python "$comfy_python" \
  "${doctor_profiles[@]}"

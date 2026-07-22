#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
COMFY_PYTHON="$SCRIPT_DIR/.slide_pipeline/ComfyUI/.venv/bin/python"

if [[ -x "$COMFY_PYTHON" ]]; then
  RUNNER_PYTHON="$COMFY_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  RUNNER_PYTHON="$(command -v python3)"
else
  printf '%s\n' 'Python 3 is required to launch the ComfyUI workflow.' >&2
  exit 1
fi

if [[ "${1:-}" == "--simple" || "${1:-}" == "simple" ]]; then
  shift
  exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" simple "$@"
fi

if [[ "${1:-}" == "comfy" || "${1:-}" == "comfy-batch" || "${1:-}" == "comfy-benchmark" || "${1:-}" == "comfy-doctor" || "${1:-}" == "comfy-ui" ]]; then
  COMMAND="$1"
  shift
  if [[ "$COMMAND" == "comfy-doctor" ]]; then
    exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" doctor "$@"
  fi
  if [[ "$COMMAND" == "comfy-ui" ]]; then
    exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" serve "$@"
  fi
  if [[ "$COMMAND" == "comfy-batch" ]]; then
    exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" batch "$@"
  fi
  if [[ "$COMMAND" == "comfy-benchmark" ]]; then
    exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" benchmark "$@"
  fi
  exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" run "$@"
fi

exec "$RUNNER_PYTHON" "$SCRIPT_DIR/comfy_restore.py" "$@"

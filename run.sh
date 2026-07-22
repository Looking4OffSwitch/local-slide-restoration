#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PYTHON="$SCRIPT_DIR/.slide_pipeline/venv/bin/python"

if [[ "${1:-}" == "comfy" || "${1:-}" == "comfy-batch" || "${1:-}" == "comfy-benchmark" || "${1:-}" == "comfy-doctor" || "${1:-}" == "comfy-ui" ]]; then
  COMMAND="$1"
  shift
  if [[ -x "$PYTHON" ]]; then
    RUNNER_PYTHON="$PYTHON"
  elif command -v python3 >/dev/null 2>&1; then
    RUNNER_PYTHON="$(command -v python3)"
  else
    printf '%s\n' 'Python 3 is required to launch the ComfyUI workflow.' >&2
    exit 1
  fi
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

if [[ ! -x "$PYTHON" ]]; then
  printf 'Pipeline is not installed. Run %s/setup_pipeline.sh first.\n' "$SCRIPT_DIR" >&2
  exit 1
fi

if [[ "${1:-}" == "doctor" || "${1:-}" == "benchmark" || "${1:-}" == "run" ]]; then
  COMMAND="$1"
  shift
  exec "$PYTHON" "$SCRIPT_DIR/slide_pipeline.py" "$COMMAND" "$@"
fi

exec "$PYTHON" "$SCRIPT_DIR/slide_pipeline.py" run "$@"

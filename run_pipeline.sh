#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PYTHON="$SCRIPT_DIR/.slide_pipeline/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Pipeline is not installed. Run %s/setup_pipeline.sh first.\n' "$SCRIPT_DIR" >&2
  exit 1
fi

if [[ "${1:-}" == "doctor" ]]; then
  shift
  exec "$PYTHON" "$SCRIPT_DIR/slide_pipeline.py" doctor "$@"
fi

exec "$PYTHON" "$SCRIPT_DIR/slide_pipeline.py" run "$@"

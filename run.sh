#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/code/rag_env/Scripts/python.exe"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: Python executable not found in rag_env at $VENV_PYTHON"
  echo "Please create or activate the virtual environment in code/rag_env."
  exit 1
fi

echo "Installing pinned requirements into rag_env..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/code/requirements.txt"

echo "Running the main agent..."
cd "$ROOT_DIR/code"
"$VENV_PYTHON" main.py "$@"
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[INFO] Installing Python dependencies..."
uv sync

echo "[INFO] Downloading robot assets..."
"${REPO_ROOT}/scripts/download_assets.sh"

echo "[INFO] Setup complete."

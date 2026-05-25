#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[INFO] Installing Python dependencies..."
pip install --no-cache-dir -r "${REPO_ROOT}/requirements.txt"
pip install --no-cache-dir --no-deps -e "${REPO_ROOT}"

echo "[INFO] Downloading robot assets..."
"${REPO_ROOT}/scripts/download_assets.sh"

echo "[INFO] Setup complete."

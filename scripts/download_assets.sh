#!/usr/bin/env bash
# Download Dodo (dodo_daimao) robot assets from HuggingFace.
# Usage: ./scripts/download_assets.sh [asset_dir]
set -euo pipefail

ASSET_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/assets/robots/dodo}"
HF_REPO="https://huggingface.co/ultravanish/dodo-rl-checkpoints/resolve/main"

_hf_download() {
    local src="$1"
    local dst="$2"
    if [ -f "${dst}" ]; then
        echo "[SKIP] $(basename ${dst}) already exists"
    else
        echo "[DOWN] $(basename ${dst})"
        wget -q --show-progress --continue -O "${dst}" "${src}"
    fi
}

# URDF + joint CSV (primary files needed by Genesis)
echo "[INFO] Downloading Dodo URDF assets..."
mkdir -p "${ASSET_DIR}/urdf"
_hf_download "${HF_REPO}/urdf/dodo_daimao.urdf" "${ASSET_DIR}/urdf/dodo_daimao.urdf"
_hf_download "${HF_REPO}/urdf/dodo_daimao.csv"  "${ASSET_DIR}/urdf/dodo_daimao.csv"

# Mesh files (STL) — referenced by the URDF
echo "[INFO] Downloading Dodo mesh assets..."
mkdir -p "${ASSET_DIR}/meshes"
for mesh in body hip_left hip_right upper_leg_left upper_leg_right \
            lower_leg_left lower_leg_right foot_left foot_right \
            foot_sole_left foot_sole_right; do
    _hf_download "${HF_REPO}/meshes/${mesh}.STL" "${ASSET_DIR}/meshes/${mesh}.STL"
done

# Joint config (joint order for the controller)
echo "[INFO] Downloading Dodo config assets..."
mkdir -p "${ASSET_DIR}/config"
_hf_download "${HF_REPO}/config/joint_names_dodo_daimao.yaml" "${ASSET_DIR}/config/joint_names_dodo_daimao.yaml"

echo "[INFO] All Dodo assets saved to ${ASSET_DIR}"

#!/usr/bin/env bash
# Launch a training run on a Vast.ai instance.
# Usage: ./scripts/vast_launch.sh <instance_ip> <run_name> [extra docker args]
set -euo pipefail

INSTANCE_IP="${1:?Usage: $0 <instance_ip> <run_name>}"
RUN_NAME="${2:?Usage: $0 <instance_ip> <run_name>}"
EXTRA="${3:-}"

IMAGE="dodo-rl-genesis:latest"

echo "==> Syncing code to ${INSTANCE_IP}..."
rsync -az --exclude='.git' --exclude='checkpoints' --exclude='runs' \
  ./ "root@${INSTANCE_IP}:/workspace/"

echo "==> Building image on remote..."
ssh "root@${INSTANCE_IP}" "cd /workspace && docker build -f docker/Dockerfile -t ${IMAGE} ."

echo "==> Starting training run '${RUN_NAME}'..."
ssh "root@${INSTANCE_IP}" "cd /workspace && docker run --gpus all --rm \
  -v /workspace/checkpoints:/workspace/checkpoints \
  -v /workspace/runs:/workspace/runs \
  -e WANDB_API_KEY=${WANDB_API_KEY:-} \
  ${EXTRA} \
  ${IMAGE} \
  python scripts/train.py run_name=${RUN_NAME}"

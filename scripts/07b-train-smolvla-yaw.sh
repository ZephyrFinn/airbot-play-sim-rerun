#!/usr/bin/env bash
# 微调 SmolVLA。所有权重都用本地目录，全程离线，不碰网络。
set -e
SIMRUN=/home/tz/workspace/airbot/sim-rerun
export PATH="$SIMRUN/ws/venv-vla/bin:$PATH"
export HF_HOME="$SIMRUN/ws/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # 权重已在本地，强制离线
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 降显存碎片，几乎零代价

STEPS=${STEPS:-12000}
BATCH=${BATCH:-24}
OUT=${OUT:-$SIMRUN/artifacts/train_smolvla_yaw}

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path="$SIMRUN/ws/smolvla_base" \
  --policy.vlm_model_name="$SIMRUN/ws/smolvlm2_base" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/airbot_sim_pick_place_yaw \
  --dataset.root="$SIMRUN/artifacts/ds_pick_place_yaw" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=4000 \
  --log_freq=100 \
  --output_dir="$OUT" \
  --wandb.enable=false \
  "$@"

#!/usr/bin/env bash
# 大样本评测：把 n 从 40 提到 200，才能分辨几个百分点的差别。
SIMRUN=/home/tz/workspace/airbot/sim-rerun
cd "$SIMRUN/ws/auto-atomic-operation"
export PATH="$SIMRUN/ws/venv-vla/bin:$PATH"
export HF_HOME=$SIMRUN/ws/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 MUJOCO_GL=egl
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
N=${N:-200}
for C in 008000 012000; do
  echo "### $(date -Is) ckpt $C  n=$N"
  python 06-eval-policy.py \
    --checkpoint "$SIMRUN/artifacts/train_smolvla/checkpoints/$C/pretrained_model" \
    --episodes "$N" \
    --out "$SIMRUN/artifacts/eval_big_$C.json"
  echo "### ckpt $C done exit=$?"
done
echo "### $(date -Is) 专家基线 n=$N"
python 12-expert-baseline.py --episodes "$N" --out "$SIMRUN/artifacts/expert_baseline_big.json"
echo "### ALL_BIG_EVAL_DONE $(date -Is)"

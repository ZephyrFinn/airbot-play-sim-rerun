#!/usr/bin/env bash
# 泛化边界扫描：把物块位置逐步推出训练范围（±3cm），看模型什么时候崩。
# 每个范围都同时跑【专家】和【v1 模型】——专家给出该范围的成功率上界，
# 没有这个上界，就分不清"模型不行"还是"任务本身变难了"。
SIMRUN=/home/tz/workspace/airbot/sim-rerun
cd "$SIMRUN/ws/auto-atomic-operation"
export PATH="$SIMRUN/ws/venv-vla/bin:$PATH"
export HF_HOME=$SIMRUN/ws/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 MUJOCO_GL=egl
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
N=${N:-100}
CKPT="$SIMRUN/artifacts/train_smolvla/checkpoints/012000/pretrained_model"
for R in 03 05 07 09; do
  echo "######## $(date -Is)  范围 ±0.${R} m  ########"
  echo "--- 专家（上界）---"
  python 12-expert-baseline.py --config-name "pick_and_place_r$R" --episodes "$N" \
      --out "$SIMRUN/artifacts/ood_expert_r$R.json" 2>&1 | tail -6
  echo "--- v1 模型 ---"
  python 06-eval-policy.py --checkpoint "$CKPT" --config-name "pick_and_place_r$R" \
      --episodes "$N" --out "$SIMRUN/artifacts/ood_v1_r$R.json" 2>&1 | tail -8
done
echo "### OOD_SWEEP_DONE $(date -Is)"

#!/usr/bin/env bash
# 等 VLM 权重下载完 → 校验 sha256 → 自动开始训练
SIMRUN=/home/tz/workspace/airbot/sim-rerun
W="$SIMRUN/ws/smolvlm2_base/model.safetensors"
EXPECT=b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3
SIZE=2029990624

echo "### $(date -Is) 等待 VLM 权重下载完成..."
while true; do
  s=$(stat -c%s "$W" 2>/dev/null || echo 0)
  if [ "$s" -eq "$SIZE" ] && ! pgrep -f "curl.*SmolVLM2" >/dev/null; then break; fi
  sleep 30
done

echo "### $(date -Is) 大小已达 $SIZE，校验 sha256..."
ACTUAL=$(sha256sum "$W" | cut -d' ' -f1)
echo "期望 $EXPECT"
echo "实际 $ACTUAL"
if [ "$ACTUAL" != "$EXPECT" ]; then
  echo "### 校验失败，停止。不要用坏权重训练。"
  exit 1
fi
echo "### 校验通过 ✓"

echo "### $(date -Is) 开始训练 SmolVLA"
exec "$SIMRUN/scripts/07-train-smolvla.sh"

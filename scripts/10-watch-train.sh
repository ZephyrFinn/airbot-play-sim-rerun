#!/usr/bin/env bash
# 从训练日志里剥出干净的 metrics 行（去掉 tqdm 进度条），并打印最近几条
L=/home/tz/workspace/airbot/sim-rerun/logs/23-train-smolvla.log
O=/home/tz/workspace/airbot/sim-rerun/logs/23b-train-metrics.log
grep -a "ot_train.py:641" "$L" | sed 's/.*ot_train.py:641 //' > "$O"
echo "记录点: $(wc -l < "$O")"
tail -${1:-5} "$O" | awk '{printf "%-12s %-14s %-14s %-12s %s\n", $1, $5, $6, $7, $10}'

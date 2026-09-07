#!/usr/bin/env python3
"""脚本化专家在同样随机种子上的成功率 —— VLA 成功率的对照上界。
用和 06-eval-policy.py 完全相同的 seed 序列，保证两组数字可比。
"""
import argparse, json
import numpy as np
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import TaskRunner
from auto_atom import load_task_file_hydra

ap = argparse.ArgumentParser()
ap.add_argument("--config-name", default="pick_and_place")
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--max-updates", type=int, default=400)
ap.add_argument("--out", default="/home/tz/workspace/airbot/sim-rerun/artifacts/expert_baseline.json")
a = ap.parse_args()

from collections import Counter
succ, fails, per = [], Counter(), []
seed = 5000
while len(succ) < a.episodes:
    seed += 1
    tf = load_task_file_hydra(a.config_name,
                              overrides=[f"task.seed={seed}", "+env.viewer.disable=true"])
    r = TaskRunner().from_config(tf)
    try:
        upd = r.reset()
        step = 0
        while step < a.max_updates:
            upd = r.update(); step += 1
            if bool(np.all(upd.done)): break
        s = [bool(x) for x in np.asarray(upd.success).reshape(-1)]
        f = [rec.stage_name for rec in r.records if rec.status.value != "succeeded"]
        for x in f: fails[x] += 1
        succ += s; per.append({"seed": seed, "success": s, "steps": step, "failed_stages": f})
        print(f"[{len(succ)}/{a.episodes}] seed={seed} {s} steps={step}", flush=True)
    finally:
        r.close()

out = {"策略": "ConfigDrivenDemoPolicy（脚本化专家）",
       "episodes": len(succ), "成功数": sum(succ),
       "success_rate": round(sum(succ)/len(succ), 4),
       "failed_stage_counts": dict(fails)}
print("\n" + json.dumps(out, indent=2, ensure_ascii=False))
json.dump({"summary": out, "per_episode": per}, open(a.out, "w"), indent=2, ensure_ascii=False)
print("已写入", a.out)

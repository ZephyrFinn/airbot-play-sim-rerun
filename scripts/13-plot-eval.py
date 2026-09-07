#!/usr/bin/env python3
"""VLA vs 脚本化专家：成功率 + 步数分布对比。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

vla = json.load(open("artifacts/train_smolvla/checkpoints/004000/eval_result.json"))
exp = json.load(open("artifacts/expert_baseline.json"))

def steps_by_outcome(d):
    ok, bad = [], []
    for e in d["per_episode"]:
        for s in e["success"]:
            (ok if s else bad).append(e["steps"])
    return ok, bad

v_ok, v_bad = steps_by_outcome(vla)
e_ok, e_bad = steps_by_outcome(exp)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# 1) 成功率
names = ["脚本化专家\n(上界)", "SmolVLA\n(ckpt 4000)"]
rates = [exp["summary"]["success_rate"], vla["summary"]["success_rate"]]
b = ax[0].bar(names, rates, color=["#4caf50", "#4c9be8"], width=.55)
for i, r in enumerate(rates):
    n = exp["summary"] if i == 0 else vla["summary"]
    ax[0].text(i, r + .02, f"{r*100:.1f}%\n{n['成功数']}/{n['episodes']}", ha="center", fontsize=10)
ax[0].set_ylim(0, 1.18); ax[0].set_ylabel("成功率")
ax[0].set_title("闭环成功率（同一批随机种子 5001-5020）")
ax[0].grid(alpha=.3, axis="y")

# 2) 步数分布
ax[1].hist([e_ok, v_ok], bins=12, label=["专家(成功)", "VLA(成功)"],
           color=["#4caf50", "#4c9be8"])
if v_bad:
    ax[1].axvline(400, color="#e53935", ls="--", lw=1.6)
    ax[1].text(400, ax[1].get_ylim()[1]*.72, f" {len(v_bad)} 次撞上限\n (max_updates=400)",
               color="#e53935", fontsize=8.5, va="top")
ax[1].set_xlabel("完成步数"); ax[1].set_ylabel("次数")
ax[1].set_title("完成步数分布\n专家极稳(245-254)，VLA 方差大")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3, axis="y")

# 3) 失败类型
ax[2].axis("off")
txt = (
 "失败归因\n"
 "─────────────────────────────\n"
 f"VLA 失败 {len(v_bad)} 次，全部是：\n"
 "   steps = 400（撞 max_updates 上限）\n"
 "   failed_stage_counts = {}（空）\n\n"
 "→ 是【超时】不是【stage 判定失败】。\n"
 "  策略没做错动作，是没在预算内完成。\n\n"
 "对比专家：40/40 成功，步数 245~254，\n"
 "  极差只有 9 步 —— 脚本化策略的确定性。\n\n"
 "VLA 成功的集中位数只有 178 步，比专家还快，\n"
 "说明它学到的是一条更直接的路径；\n"
 "但一旦初始位姿落在训练分布边缘就彻底卡住。\n\n"
 "→ 典型的小数据模仿学习症状：\n"
 "   分布内很好，分布边缘断崖。"
)
ax[2].text(0, 1, txt, va="top", ha="left", fontsize=9.5)

fig.suptitle("SmolVLA 闭环评测 vs 脚本化专家基线 · MuJoCo pick_and_place", fontsize=13)
fig.tight_layout()
fig.savefig("artifacts/eval_comparison.png", dpi=130)
print(json.dumps({
    "专家": exp["summary"], "VLA": vla["summary"],
    "专家成功步数": {"min": min(e_ok), "中位": int(np.median(e_ok)), "max": max(e_ok)},
    "VLA成功步数": {"min": min(v_ok), "中位": int(np.median(v_ok)), "max": max(v_ok)},
    "VLA失败步数": sorted(set(v_bad)),
}, ensure_ascii=False, indent=2))

#!/usr/bin/env python3
"""把 Part A 的两个核心实验画成一张图，面试可以直接展示。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

J = [f"j{i}" for i in range(1, 7)]
before      = [0.0]*6
after_plan  = [0.0]*6
after_exec  = [0.499235, -0.299105, 0.399757, 0.000372, 0.200108, 0.000727]
target      = [0.5, -0.3, 0.4, 0.0, 0.2, 0.0]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

# 图1: Plan vs Execute 的关节角
x = np.arange(6)
ax[0].axhline(0, color="#cfd4da", lw=1)
ax[0].bar(x, after_exec, 0.5, color="#e8734c", label="after Plan & Execute", zorder=2)
# before / after Plan 全是 0，柱子看不见 -> 用标记画在零线上，并标注
ax[0].plot(x, before,     "s", ms=9, color="#9aa5b1", label="before (全 0)",      zorder=3)
ax[0].plot(x, after_plan, "o", ms=5, color="#4c9be8", label="after Plan (仍全 0)", zorder=4)
ax[0].plot(x, target, "k*", ms=13, linestyle="none", label="target", zorder=5)
ax[0].annotate("Plan 之后六个关节全部仍为 0\n（两组标记完全重合在零线上）",
               xy=(3.0, 0.0), xytext=(1.6, 0.30), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="#4c9be8", lw=1.2), color="#2b6cb0")
ax[0].set_xticks(x); ax[0].set_xticklabels(J)
ax[0].set_ylim(-0.42, 0.62)
ax[0].set_ylabel("joint position (rad)")
ax[0].set_title("Plan 不改变关节状态，Plan & Execute 才改变\n协议上的差别只有 planning_options.plan_only 一个布尔量")
ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=.3, axis="y")

# 图2: 到位误差
err = [abs(a-b) for a, b in zip(after_exec, target)]
ax[1].bar(x, err, color="#e8734c")
ax[1].set_xticks(x); ax[1].set_xticklabels(J)
ax[1].set_ylabel("|实际 - 目标|  (rad)")
ax[1].set_title(f"Execute 到位误差 (max {max(err):.1e} rad)\nmock 硬件无动力学，残差来自插值+goal tolerance")
ax[1].grid(alpha=.3, axis="y")

# 图3: 失败用例的耗时对比 —— "快速失败 vs 成功"
cases  = ["正常目标\n(成功)", "越关节限位\n(失败)", "加障碍物\n(失败)", "移除障碍后\n(成功)"]
times  = [0.0056, 0.083, 0.012, 0.022]
codes  = [1, 99999, 99999, 1]
colors = ["#4caf50" if c == 1 else "#e53935" for c in codes]
b = ax[2].bar(range(4), times, color=colors)
for i, (t, c) in enumerate(zip(times, codes)):
    ax[2].text(i, t*1.05, f"{t*1000:.0f}ms\ncode={c}", ha="center", fontsize=8)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(cases, fontsize=8)
ax[2].set_ylabel("规划耗时 (s)"); ax[2].set_ylim(0, 0.105)
ax[2].set_title("快速失败 ≠ 慢速失败\n远小于 allowed_planning_time(3s) = 目标非法")
ax[2].grid(alpha=.3, axis="y")

fig.suptitle("AIRBOT Play · MoveIt 2 仿真实测（mock_components/GenericSystem）", fontsize=13)
fig.tight_layout()
fig.savefig("artifacts/partA_experiments.png", dpi=130)
print("已保存 artifacts/partA_experiments.png")

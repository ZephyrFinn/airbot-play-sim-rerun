#!/usr/bin/env python3
"""姿态随机化根因追查：四组专家对照 + 数据集 action 标准差前后对比。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

cases = [
    ("原配置\nyaw 关闭", "artifacts/expert_baseline.json", "#4caf50"),
    ("打开 yaw\n姿态硬编码", "artifacts/expert_yaw_baseline.json", "#e53935"),
    ("打开 yaw\n+relative rotation\n(猜的改法)", "artifacts/expert_yaw_adapt.json", "#9e9e9e"),
    ("打开 yaw\n+reference: object\n(正解)", "artifacts/expert_yaw_objref.json", "#4caf50"),
]
rates, labels, colors, notes = [], [], [], []
for lb, f, c in cases:
    d = json.load(open(f))["summary"]
    rates.append(d["success_rate"]); labels.append(lb); colors.append(c)
    fs = d["failed_stage_counts"]
    notes.append(f"{d['成功数']}/{d['episodes']}\n" + ("超时" if (not fs and d['success_rate'] < 1)
                 else (",".join(f"{k}×{v}" for k, v in fs.items()) if fs else "")))

old = json.load(open("artifacts/ds_pick_place/meta/stats.json"))["action"]["std"]
new = json.load(open("artifacts/ds_pick_place_yaw/meta/stats.json"))["action"]["std"]
dims = ["x", "y", "z", "qx", "qy", "qz", "qw", "grip"]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

b = ax[0].bar(range(4), rates, color=colors, width=.6)
for i, (r, n) in enumerate(zip(rates, notes)):
    ax[0].text(i, r + .04, f"{r*100:.0f}%", ha="center", fontsize=11, weight="bold")
    # 注解统一放在柱子上方、百分比之下，避免和 0% 的柱子重叠
    ax[0].text(i, r + .015, n.replace(chr(10), "  "), ha="center", fontsize=7.2, color="#555")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=8.5)
ax[0].set_ylim(0, 1.30); ax[0].set_ylabel("脚本化专家成功率")
ax[0].set_title("追查「4 维常量」的根因：四组对照\n物块会转之后，专家的硬编码姿态就抓不住了")
ax[0].grid(alpha=.3, axis="y")
ax[0].text(1, 0.62, "失败是真的抓不住\n(pick_source×8)", ha="center", fontsize=8, color="#e53935")
ax[0].text(2, 0.30, "全部超时\n改法把位姿弄坏了", ha="center", fontsize=8, color="#616161")

x = np.arange(8); w = .38
ax[1].bar(x - w/2, old, w, label="原数据集（yaw 关闭）", color="#bdbdbd")
ax[1].bar(x + w/2, new, w, label="新数据集（yaw ±30°）", color="#4c9be8")
for i in [3, 4, 5, 6]:
    ax[1].annotate("", xy=(i + w/2, new[i]), xytext=(i - w/2, 0.004),
                   arrowprops=dict(arrowstyle="->", color="#e8734c", lw=1.4))
ax[1].set_xticks(x); ax[1].set_xticklabels(dims)
ax[1].set_ylabel("action 各维度标准差")
ax[1].set_title("修复效果：四元数四维从 std=0 变成有方差\n模型这才可能学到姿态控制")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
ax[1].text(4.5, max(new)*.72, "原来这四维恒为\n[0, .7071, 0, .7071]\n→ 训练时是白送的 loss",
           fontsize=8.5, color="#e8734c", ha="center")

fig.suptitle("姿态随机化根因追查 · 从「观察到异常」到「定位并修复」", fontsize=13)
fig.tight_layout(); fig.savefig("artifacts/yaw_experiment.png", dpi=130)
print(json.dumps({"专家成功率": dict(zip([l.replace(chr(10),' ') for l in labels], rates)),
                  "旧 action std": [round(v,5) for v in old],
                  "新 action std": [round(v,5) for v in new]}, ensure_ascii=False, indent=2))

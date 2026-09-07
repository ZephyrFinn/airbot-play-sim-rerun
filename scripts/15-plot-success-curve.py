#!/usr/bin/env python3
"""训练步数 → 闭环成功率曲线，带 Wilson 95% 置信区间。

为什么必须画置信区间：n=40 时，100% 和 97.5% 只差 1 集，
不加区间就会把噪声当成"过拟合"来解读。这是最容易在面试里被追问的地方。
"""
import json, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score 区间：小样本比例的正确算法。
    普通的正态近似 p±z*sqrt(p(1-p)/n) 在 p=1 时区间宽度为 0，明显是错的。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c-h), min(1.0, c+h)


pts = []
for step, c in [(4000, "004000"), (8000, "008000"), (12000, "012000")]:
    d = json.load(open(f"artifacts/train_smolvla/checkpoints/{c}/eval_result.json"))
    s = d["summary"]
    ok = [x for e in d["per_episode"] for x in e["success"]]
    steps_ok = [e["steps"] for e in d["per_episode"] for x in e["success"] if x]
    lo, hi = wilson(s["成功数"], s["episodes"])
    pts.append(dict(step=step, k=s["成功数"], n=s["episodes"], rate=s["success_rate"],
                    lo=lo, hi=hi, med_steps=int(np.median(steps_ok))))

exp = json.load(open("artifacts/expert_baseline.json"))["summary"]
e_lo, e_hi = wilson(exp["成功数"], exp["episodes"])

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

x = [p["step"] for p in pts]; y = [p["rate"] for p in pts]
lo = [p["rate"]-p["lo"] for p in pts]; hi = [p["hi"]-p["rate"] for p in pts]
ax[0].errorbar(x, y, yerr=[lo, hi], fmt="o-", color="#4c9be8", capsize=6,
               lw=2, ms=9, label="SmolVLA（Wilson 95% CI）")
ax[0].axhline(exp["success_rate"], color="#4caf50", ls="--", lw=2, label="脚本化专家 = 100%（上界）")
ax[0].fill_between([3000, 13000], e_lo, 1.0, color="#4caf50", alpha=.10)
for p in pts:
    ax[0].annotate(f"{p['k']}/{p['n']}", (p["step"], p["rate"]),
                   textcoords="offset points", xytext=(0, -20), ha="center", fontsize=9)
ax[0].set_xlim(3000, 13000); ax[0].set_ylim(0.55, 1.06)
ax[0].set_xticks(x); ax[0].set_xlabel("训练步数"); ax[0].set_ylabel("闭环成功率")
ax[0].set_title("训练步数 → 成功率（每点 40 集，同一批 seed）")
ax[0].legend(fontsize=9, loc="lower right"); ax[0].grid(alpha=.3)

ax[1].axis("off")
txt = [
 "怎么读这张图", "─"*44, "",
 "1. 4000 → 8000 步是【真提升】",
 f"   87.5% [{pts[0]['lo']*100:.0f}%, {pts[0]['hi']*100:.0f}%]  →  100% [{pts[1]['lo']*100:.0f}%, 100%]",
 "   区间几乎不重叠，这个提升不是噪声。", "",
 "2. 8000 → 12000 的回落【不能算过拟合】",
 f"   100% vs 97.5%，只差 1 集（40 集里）。",
 f"   区间 [{pts[2]['lo']*100:.0f}%, {pts[2]['hi']*100:.0f}%] 与 8000 完全重叠。",
 "   n=40 分辨不出 2.5 个百分点的差别。", "",
 "3. 为什么用 Wilson 而不是 p ± z·√(p(1-p)/n)",
 "   正态近似在 p=1 时给出宽度为 0 的区间，",
 "   等于宣称「100% 绝对可靠」，显然是错的。",
 f"   Wilson 给出 [{pts[1]['lo']*100:.1f}%, 100%]，才是诚实的说法。", "",
 "4. 要区分 2.5 个百分点，需要多少集？",
 "   粗略地 n ≈ 数百量级。所以下一步不是继续训，",
 "   是【把评测集扩大】或【换更难的任务】。",
]
ax[1].text(0, 1, "\n".join(txt), va="top", ha="left", fontsize=9.6)

fig.suptitle("SmolVLA 训练步数 vs 闭环成功率 · MuJoCo pick_and_place", fontsize=13)
fig.tight_layout(); fig.savefig("artifacts/success_curve.png", dpi=130)
print(json.dumps({"points": pts, "expert": exp,
                  "expert_wilson": [round(e_lo,4), round(e_hi,4)]},
                 ensure_ascii=False, indent=2))

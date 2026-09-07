#!/usr/bin/env python3
"""从 lerobot-train 的日志里抠出 loss / grad_norm / lr，画成一张图。
训练日志格式形如：
  step:1K smpl:24K ep:96 epch:1.60 loss:0.123 grdn:1.234 lr:1.0e-04 updt_s:0.123 data_s:0.001
"""
import re, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"]=["Noto Sans CJK SC","Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"]=False

LOG_FREQ = 100          # 与 --log_freq 保持一致
log = sys.argv[1] if len(sys.argv) > 1 else "logs/23-train-smolvla.log"
out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/train_loss.png"

def num(tok):
    """把 1K / 24K / 1.5M 这类缩写转成数字"""
    tok = tok.strip()
    m = {"K": 1e3, "M": 1e6, "B": 1e9}
    if tok and tok[-1] in m:
        return float(tok[:-1]) * m[tok[-1]]
    return float(tok)

steps, loss, grdn, lr = [], [], [], []
pat = re.compile(r"step:(\S+).*?loss:(\S+).*?grdn:(\S+).*?lr:(\S+)")
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if not m:
        continue
    try:
        loss.append(float(m.group(2)))
        grdn.append(float(m.group(3))); lr.append(float(m.group(4)))
        # ★ 不能直接用 step 字段：lerobot 把它打成缩写（5100 和 6000 都写作 "5K"/"6K"），
        #   直接 parse 会让多个点落在同一个 x 上，画出来是假的阶梯。
        #   log_freq=100，所以真实步数 = 记录序号 × 100。
        steps.append(len(loss) * LOG_FREQ)
    except ValueError:
        continue

if not steps:
    print("日志里还没有 loss 记录"); sys.exit(1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(steps, loss); ax[0].set_title("loss"); ax[0].set_xlabel("step"); ax[0].set_yscale("log")
ax[1].plot(steps, grdn, color="tab:orange"); ax[1].set_title("grad norm"); ax[1].set_xlabel("step")
ax[2].plot(steps, lr, color="tab:green"); ax[2].set_title("learning rate"); ax[2].set_xlabel("step")
for a in ax: a.grid(alpha=.3)
fig.suptitle(f"SmolVLA finetune on airbot_sim_pick_place  ({len(steps)} 个记录点, 末步 {int(steps[-1])})")
fig.tight_layout(); fig.savefig(out, dpi=120)
print(json.dumps({
    "记录点": len(steps), "末步": int(steps[-1]),
    "首个loss": loss[0], "末个loss": loss[-1],
    "最小loss": min(loss), "末10点均值": round(sum(loss[-10:])/len(loss[-10:]), 5),
    "图": out}, ensure_ascii=False, indent=2))

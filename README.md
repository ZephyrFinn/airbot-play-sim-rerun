# AIRBOT Play 仿真复现 — 目录索引

把 2026-07 求之科技实训的**真机**链路，在**无硬件**条件下重跑一遍，
所有数字都来自本机实跑，`logs/` 里保留了每条命令的原始输出。

## 文档

| 文件 | 内容 |
| --- | --- |
| **[仿真复现手册.md](仿真复现手册.md)** | 复现指南：每条命令、背后逻辑、输出怎么读 |
| [notes/排错手册.md](notes/排错手册.md) | 分层排查法 + 全部踩过的坑（现象→报错原文→排查→根因→解法） |
| [notes/实测数据速查.md](notes/实测数据速查.md) | 全部实测数字 + 日志出处，一页纸 |
| [notes/版本快照.md](notes/版本快照.md) | 组件精确版本 + commit 哈希 |

## 工作环节

### Part A — MoveIt 2 仿真链路
装 MoveIt 2 → colcon build → 启动 demo → 验收 → 三组量化实验

| 实验 | 结果 |
| --- | --- |
| Plan vs Plan&Execute | Plan 后关节全 0；Execute 到位，误差 ≤8.9e-4 rad |
| 越关节限位 | error 99999，**83 ms** 快速失败（远小于 3s 上限 = 目标非法） |
| 碰撞检测（三次对照） | 无障碍 ✓(14ms) → 加障碍 ✗(12ms) → 移除 ✓(22ms) |

真机分支做了静态复核（无硬件），`#!/usr/bin/python3` 那个坑原样复现。

### Part B — VLA 仿真链路
MuJoCo 采集 → LeRobot 数据集 → SmolVLA 微调 → 闭环评测

| | 成功率 | Wilson 95% CI | 成功时中位步数 |
| --- | --- | --- | --- |
| **脚本化专家（上界）** | **40/40 = 100%** | [91.2%, 100%] | 249 |
| SmolVLA @4000 步 | 35/40 = 87.5% | [74.0%, 94.5%] | 176 |
| **SmolVLA @8000 步** | **40/40 = 100%** | [91.2%, 100%] | 185 |
| SmolVLA @12000 步 | 39/40 = 97.5% | [87.1%, 99.6%] | 177 |

- 数据集：**60 集 / 14954 帧**，8 维 state+action、双路 256×256 视频
- 训练：450M 参数（100M 可训练），12000 步，loss 1089 → 0.019，显存 6.3 GB
- 失败形态：**全部是超时卡死**（`steps=400`、`failed_stage_counts` 为空），不是动作做错

> **这个 100% 有天花板**：4/8 动作维度是常量（专家不转手腕）、
> 评测与训练同分布、数据里没有纠错行为。详见手册 B.8。

## 目录

```
sim-rerun/
├── 仿真复现手册.md      ← 主文档
├── env.sh                统一路径变量
├── scripts/
│   ├── 00-shell.sh              每个新终端的三层环境加载（自动判断 zsh/bash）
│   ├── 01-sim-demo.sh           启动 MoveIt 2 仿真
│   ├── 02-plan-vs-execute.py    Plan vs Plan&Execute 对照实验
│   ├── 03-negative-tests.py     反面用例（越限位 / 碰撞）
│   ├── 04-dump-obs-keys.py      打印 MuJoCo 观测字典的特征契约
│   ├── 05-collect-lerobot-dataset.py   采集示教数据 → LeRobot 数据集
│   ├── 06-eval-policy.py        闭环评测策略成功率
│   ├── 07-train-smolvla.sh      微调 SmolVLA（全离线）
│   └── geturls.py               从镜像索引解析 wheel 直链（多线程下载用）
├── logs/                 所有命令的原始输出（留痕，别删）
├── artifacts/            截图、视频、数据集
│   ├── rviz_sim.png
│   ├── sim_pick_and_place.mp4
│   └── ds_pick_place/           60 集 LeRobot 数据集
├── notes/                六份笔记
└── ws/
    ├── AIRBOT-Play-Hardware-with-Moveit2/   MoveIt 2 工作空间 (feature/jazzy)
    ├── auto-atomic-operation/               MuJoCo 仿真 (aao)
    ├── venv-vla/                            Python 环境
    ├── smolvla_base/                        SmolVLA 动作专家权重
    └── smolvlm2_base/                       SmolVLM2 视觉语言骨干
```

## 主要结论

**1. 失败耗时可以定位到规划管线的哪一环。** `allowed_planning_time` 为 3 s，
但起始态碰撞在 12 ms 被 `PlanningRequestAdapter 'CheckStartStateCollision'` 拒绝、
越关节限位在 83 ms 因 OMPL 采不到合法状态而放弃——两者都远未用满预算，
说明失败发生在搜索之前。耗满超时才是"搜索空间里真的没有解"。

**2. SRDF 的碰撞检测需要外部障碍物才能验证。** 30 多条 `disable_collisions`
关掉了相邻连杆与结构上永不相碰的对，靠折叠关节很难触发自碰撞；
往 `PlanningScene` 注入 0.4³ m 障碍物后，同一目标的三次对照为
成功(14 ms) → 失败(12 ms) → 移除后成功(22 ms)。

**3. 常量维度会污染整条 loss 曲线。** 首版数据集中 8 维动作有 4 维标准差为 0，
LeRobot 归一化为 `(x-mean)/(std+1e-8)`，float32 舍入残差 `6.557e-7` 被放大至 `65.6`，
平方进 MSE 约 `4.3e3`，而有效维度归一化后仅 O(1)。
修复数据源后同模型同超参的起始 loss 由 1089.2 变为 1.627。
**结论：跨数据集比较 loss 无意义，看 loss 之前需先检查归一化统计量。**

**4. 分布内成功率不反映泛化能力。** 将物块初始位置随机范围由训练时的 ±3 cm
推至 ±9 cm，脚本化专家全程保持 100%（每档 n=100），而策略由 95% 衰减至 31%；
失败形态同时由"动作停滞超时"迁移为 `place_source` 阶段失败。
专家基线是这条曲线成立的前提——没有它无法区分模型退化与任务难度上升。

**5. 脚本化专家的适应性受限于配置的表达能力。** 打开物块偏航角随机化后专家自身
由 100% 降至 40%，根因是抓取姿态为硬编码四元数、参考系为 `object_world`（仅继承位置）；
改用 `reference: object`（继承朝向）后恢复 100%。

## 仓库不包含什么（以及怎么补回来）

为了让仓库保持在 5 MB 以内，**只提交自己的产出**：文档、脚本、原始日志、图表、视频、评测 json。
以下 19 GB 不在仓库里，都可以按手册重建：

| 缺的东西 | 体积 | 怎么补 |
| --- | --- | --- |
| `ws/auto-atomic-operation/` | 300 MB | `git clone` + `git lfs pull`，见手册 B.1 |
| `ws/AIRBOT-Play-Hardware-with-Moveit2/` | 95 MB | `git clone` + `colcon build`，见手册 A.2 |
| `ws/venv-vla/` | 6.1 GB | `uv venv` + 手册 B.1 的安装命令 |
| `ws/smolvla_base/` `ws/smolvlm2_base/` | 2.8 GB | 从 HuggingFace 拉，见手册 B.4.7 |
| `artifacts/ds_pick_place{,_yaw}/` | 141 MB | 跑 `scripts/05-collect-lerobot-dataset.py` 重采 |
| `artifacts/train_smolvla{,_yaw}/` | 9.1 GB | 跑 `scripts/07-train-smolvla.sh` 重训（约 90 分钟） |

**留在仓库里的是结论和证据**：`logs/` 是每条命令的原始输出，`artifacts/*.json` 是全部评测结果，
`artifacts/*.png` / `*.mp4` 是图表和 rollout 录像。**所有数字都能追到出处。**

## 三条命令验证环境还活着

```bash
./scripts/01-sim-demo.sh                                    # 终端 A：起仿真
. scripts/00-shell.sh && ros2 control list_controllers      # 终端 B：两个 active
python3 scripts/02-plan-vs-execute.py                       # 核心实验
```

## 日志索引

| 日志 | 步骤 |
| --- | --- |
| `01-apt-moveit.log` / `01b-versions.log` | 装 MoveIt 2 + 版本 |
| `02-colcon-build.log` | rosdep + colcon build |
| `03-pkg-verify.log` | 包可见性 |
| `04-sim-demo.log` | **仿真全量日志（排错主要 grep 这个）** |
| `05-sim-verify.log` | 控制器 / joint_states 验收 |
| `06-plan-vs-execute.log` | Plan vs Execute 实测 |
| `07*-lfs*.log` | MuJoCo 资产 LFS |
| `09/12/16/17-*.log` | Python 环境安装（含 torch 提速全过程） |
| `10-realrobot-static-review.log` | 真机分支静态复核 |
| `11-negative-tests.log` | 反面用例 |
| `13/14-aao-*.log` | MuJoCo 仿真 |
| `15-obs-keys.log` | 观测字典特征契约 |
| `18-collect-dataset.log` | 数据采集 60 集 |
| `19/20-smolvla-*.log` | SmolVLA 权重与配置 |
| `21-dataset-stats.log` | 归一化统计量（发现 4 维 std=0） |
| `22-dataset-load-check.log` | 数据集读取自检 |

## 图表与视频产出

| 文件 | 内容 |
| --- | --- |
| `artifacts/partA_experiments.png` | MoveIt 2 三组实验（Plan/误差/快速失败） |
| `artifacts/train_loss.png` | SmolVLA 训练曲线（loss / grad norm / lr） |
| `artifacts/success_curve.png` | 训练步数 → 成功率，带 Wilson 置信区间 |
| `artifacts/eval_comparison.png` | VLA vs 专家，含步数分布与失败归因 |
| `artifacts/vla_rollout_OK_seed5002.mp4` | VLA 成功案例（173 步完成） |
| `artifacts/vla_rollout_FAIL_seed7001.mp4` | VLA 失败案例（卡满 400 步超时） |
| `artifacts/sim_pick_and_place.mp4` | 脚本化专家演示 |
| `artifacts/rviz_sim.png` / `rviz_after_execute.png` | RViz 截图（执行前后姿态对比） |

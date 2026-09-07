#!/usr/bin/env python3
"""
用 aao 的 PolicyEvaluator 在 MuJoCo 里闭环评测训练好的 SmolVLA。

对照真机推理：
    真机: LeRobot policy → gRPC → airbot-arm → 电机
    仿真: LeRobot policy → evaluator.update(action) → env.apply_pose_action → MuJoCo

PolicyEvaluator 不生成动作，只做两件事：
  1. 把外部策略给的动作施加到环境
  2. 按任务定义判定每个 stage 的成功/失败
所以拿到的不是一个 0/1，而是"卡在哪个 stage、为什么失败"。

用法:
  python 06-eval-policy.py --checkpoint <ckpt目录> --episodes 30 [--config-name pick_and_place]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from auto_atom import ExecutionContext, PolicyEvaluator, TaskUpdate, load_task_file_hydra

# ── 特征契约：必须和采集时（05-collect-lerobot-dataset.py）逐字一致 ──
STATE_KEYS = ["arm/pose/position", "arm/pose/orientation", "gripper/joint_state/position"]
CAMS = {"wrist": "wrist_cam/color/image_raw", "env0": "env0_cam/color/image_raw"}
IMG_HW = (256, 256)
TASK_TEXT = "pick up the block and place it on the pedestal"


def action_applier(context: ExecutionContext, action: Any,
                   env_mask: Optional[np.ndarray] = None) -> None:
    if action is None:
        return
    context.backend.env.apply_pose_action(
        "arm", action["position"], action["orientation"], action["gripper"], kinematic=False)


def observation_getter(context: ExecutionContext) -> dict:
    return context.backend.env.capture_observation()


def _resize(img, hw):
    import cv2
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


class SmolVLAPolicyWrapper:
    """把 LeRobot 的 SmolVLA 包成 aao 期望的 policy 接口。

    SmolVLA 输出 8 维动作 = 位置(3) + 四元数(4) + 夹爪(1)，
    正好拆回 apply_pose_action 需要的三个参数。
    """

    def __init__(self, ckpt: str, device: str):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.p = SmolVLAPolicy.from_pretrained(ckpt).to(device).eval()
        # ★ 关键：策略吃的不是原始观测，而是经过 preprocessor 流水线处理过的 batch。
        #   流水线里做了三件事：语言 tokenize(observation.language.tokens)、
        #   按训练集统计量归一化、搬到目标 device。
        #   跳过它直接 select_action 会报 KeyError: 'observation.language.tokens'。
        self.pre, self.post = make_pre_post_processors(self.p.config, pretrained_path=ckpt)
        self.device = device

    def reset(self):
        self.p.reset()

    @torch.inference_mode()
    def act(self, observation: Any, update: TaskUpdate, evaluator: PolicyEvaluator) -> dict:
        obs = observation
        n = np.asarray(obs[STATE_KEYS[0]]["data"]).shape[0]
        batch = {}
        st = np.stack([
            np.concatenate([np.atleast_1d(np.asarray(obs[k]["data"], dtype=np.float32)[i].ravel())
                            for k in STATE_KEYS])
            for i in range(n)])
        batch["observation.state"] = torch.tensor(st, dtype=torch.float32, device=self.device)
        for name, key in CAMS.items():
            imgs = np.asarray(obs[key]["data"], dtype=np.uint8)
            arr = np.stack([_resize(imgs[i], IMG_HW) for i in range(n)])
            t = torch.tensor(arr, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2) / 255.0
            batch[f"observation.images.{name}"] = t
        batch["task"] = [TASK_TEXT] * n

        batch = self.pre(batch)                                  # tokenize + 归一化 + to(device)
        a = self.p.select_action(batch)
        a = self.post(a)                                         # 反归一化回真实动作量纲
        a = a.float().cpu().numpy()                              # (B, 8)
        if a.ndim == 1:
            a = a[None, :]
        pos, ori, grip = a[:, 0:3], a[:, 3:7], a[:, 7:8]
        # 四元数必须归一化再喂给仿真，否则姿态会被缩放
        nrm = np.linalg.norm(ori, axis=1, keepdims=True)
        ori = np.where(nrm > 1e-6, ori / np.maximum(nrm, 1e-6), np.array([0., .7071, 0., .7071]))
        return {"position": pos.astype(np.float32),
                "orientation": ori.astype(np.float32),
                "gripper": grip.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pick_and_place")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-updates", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicyWrapper(a.checkpoint, dev)
    print(f"策略已加载: {a.checkpoint}  device={dev}", flush=True)

    from collections import Counter
    results, succ_flat, fail_stages = [], [], Counter()
    seed = 5000
    ep = 0
    while ep < a.episodes:
        seed += 1
        task_file = load_task_file_hydra(a.config_name,
                                         overrides=[f"task.seed={seed}", "+env.viewer.disable=true"])
        ev = PolicyEvaluator(action_applier=action_applier,
                             observation_getter=observation_getter).from_config(task_file, 10)
        try:
            policy.reset()
            update = ev.reset()
            step = -1
            for step in range(a.max_updates):
                obs = ev.get_observation()
                act = policy.act(obs, update=update, evaluator=ev)
                update = ev.update(act)
                if bool(np.all(update.done)):
                    break
            summary = ev.summarize(update, max_updates=a.max_updates, updates_used=step + 1)
            s = [bool(x) for x in np.asarray(summary.final_success).reshape(-1)]
            fails = [r.stage_name for r in ev.records if r.status.value != "succeeded"]
            for f in fails:
                fail_stages[f] += 1
            succ_flat += s
            results.append({"seed": seed, "success": s, "steps": step + 1, "failed_stages": fails})
            ep += len(s)
            print(f"[{ep}/{a.episodes}] seed={seed} success={s} steps={step+1} failed_at={fails}", flush=True)
        finally:
            ev.close()

    out = {
        "checkpoint": a.checkpoint,
        "episodes": len(succ_flat),
        "success_rate": round(sum(succ_flat) / max(len(succ_flat), 1), 4),
        "成功数": sum(succ_flat),
        "failed_stage_counts": dict(fail_stages),
    }
    print("\n" + json.dumps(out, indent=2, ensure_ascii=False))
    dst = a.out or os.path.join(os.path.dirname(a.checkpoint.rstrip("/")), "eval_result.json")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": out, "per_episode": results}, open(dst, "w"), indent=2, ensure_ascii=False)
    print("已写入", dst)


main()

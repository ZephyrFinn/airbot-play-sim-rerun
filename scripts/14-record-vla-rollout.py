#!/usr/bin/env python3
"""录一段训练好的 SmolVLA 在 MuJoCo 里实际执行的视频。
和 06-eval-policy.py 同一套推理逻辑，只是每步多抓一帧相机图像。
"""
from __future__ import annotations
import argparse, os
from typing import Any, Optional
import numpy as np, torch
import imageio.v3 as iio
from auto_atom import ExecutionContext, PolicyEvaluator, TaskUpdate, load_task_file_hydra

STATE_KEYS = ["arm/pose/position", "arm/pose/orientation", "gripper/joint_state/position"]
CAMS = {"wrist": "wrist_cam/color/image_raw", "env0": "env0_cam/color/image_raw"}
VIDEO_CAM = "env1_cam/color/image_raw"      # 第三视角，看得最清楚
IMG_HW = (256, 256)
TASK_TEXT = "pick up the block and place it on the pedestal"


def action_applier(ctx: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None):
    if action is not None:
        ctx.backend.env.apply_pose_action("arm", action["position"], action["orientation"],
                                          action["gripper"], kinematic=False)


def observation_getter(ctx: ExecutionContext):
    return ctx.backend.env.capture_observation()


def _rs(img, hw):
    import cv2
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--config-name", default="pick_and_place")
ap.add_argument("--seed", type=int, default=7001)
ap.add_argument("--max-updates", type=int, default=400)
ap.add_argument("--out", default="/home/tz/workspace/airbot/sim-rerun/artifacts/vla_rollout.mp4")
a = ap.parse_args()

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
dev = "cuda" if torch.cuda.is_available() else "cpu"
policy = SmolVLAPolicy.from_pretrained(a.checkpoint).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=a.checkpoint)
print("策略已加载", flush=True)

tf = load_task_file_hydra(a.config_name, overrides=[f"task.seed={a.seed}", "+env.viewer.disable=true"])
ev = PolicyEvaluator(action_applier=action_applier, observation_getter=observation_getter).from_config(tf, 10)
frames = []
try:
    policy.reset()
    upd = ev.reset()
    for step in range(a.max_updates):
        obs = ev.get_observation()
        n = np.asarray(obs[STATE_KEYS[0]]["data"]).shape[0]
        batch = {}
        st = np.stack([np.concatenate([np.atleast_1d(np.asarray(obs[k]["data"], dtype=np.float32)[i].ravel())
                                       for k in STATE_KEYS]) for i in range(n)])
        batch["observation.state"] = torch.tensor(st, dtype=torch.float32, device=dev)
        for nm, key in CAMS.items():
            im = np.asarray(obs[key]["data"], dtype=np.uint8)
            arr = np.stack([_rs(im[i], IMG_HW) for i in range(n)])
            batch[f"observation.images.{nm}"] = torch.tensor(arr, dtype=torch.float32, device=dev).permute(0,3,1,2)/255.
        batch["task"] = [TASK_TEXT]*n
        with torch.inference_mode():
            act = post(policy.select_action(pre(batch))).float().cpu().numpy()
        if act.ndim == 1: act = act[None, :]
        pos, ori, grip = act[:,0:3], act[:,3:7], act[:,7:8]
        nr = np.linalg.norm(ori, axis=1, keepdims=True)
        ori = np.where(nr > 1e-6, ori/np.maximum(nr,1e-6), np.array([0.,.7071,0.,.7071]))
        # 抓视频帧（只取第 0 个并行环境）
        vf = obs.get(VIDEO_CAM, {}).get("data")
        if vf is None: vf = obs[CAMS["env0"]]["data"]
        frames.append(np.asarray(vf, dtype=np.uint8)[0])
        upd = ev.update({"position": pos.astype(np.float32),
                         "orientation": ori.astype(np.float32),
                         "gripper": grip.astype(np.float32)})
        if bool(np.all(upd.done)): break
    s = [bool(x) for x in np.asarray(upd.success).reshape(-1)]
    print(f"seed={a.seed} success={s} steps={len(frames)}")
finally:
    ev.close()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
iio.imwrite(a.out, np.stack(frames), fps=25, codec="libx264")
print("已保存", a.out, f"({len(frames)} 帧)")

#!/usr/bin/env python3
"""
在 MuJoCo 里用脚本化专家策略采集示教数据，直接写成 LeRobot 数据集。

这一步对应真机时的 airdc 数采：
    真机: 遥操作(lead 臂) → airdc 同步录制 → mcap → 转 LeRobot 格式
    仿真: aao 的 TaskRunner 当"专家" → 直接写 LeRobot 格式（省掉 mcap 中转）

特征契约（和真机那次完全一致的 8 维）：
    observation.state  = arm/pose/position(3) + arm/pose/orientation(4,四元数) + gripper(1)
    action             = action/arm/pose/position(3) + action/arm/pose/orientation(4) + action/gripper(1)
    observation.images.{wrist,env0} = (H,W,3) uint8

用法：
    python 05-collect-lerobot-dataset.py --config-name pick_and_place \
        +collect.episodes=40 +collect.out=/path/to/dataset
"""
import os, sys, shutil, json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import TaskRunner

STATE_KEYS = ["arm/pose/position", "arm/pose/orientation", "gripper/joint_state/position"]
ACTION_KEYS = ["action/arm/pose/position", "action/arm/pose/orientation",
               "action/gripper/joint_state/position"]
CAMS = {"wrist": "wrist_cam/color/image_raw", "env0": "env0_cam/color/image_raw"}
IMG_HW = (256, 256)          # 下采样到 256×256，原始 352×640 太大且非方形
TASK_TEXT = "pick up the block and place it on the pedestal"


def _resize(img, hw):
    import cv2
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def _vec(obs, keys, env_idx):
    """把若干个观测 key 拼成一个一维特征向量（这就是"特征契约"的具体实现）"""
    parts = []
    for k in keys:
        d = np.asarray(obs[k]["data"], dtype=np.float32)
        parts.append(np.atleast_1d(d[env_idx]))
    return np.concatenate(parts).astype(np.float32)


@hydra.main(config_path=str(get_config_dir()), config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig):
    raw = OmegaConf.to_container(cfg, resolve=True)
    col = raw.pop("collect", {}) or {}
    n_ep_target = int(col.get("episodes", 40))
    out = col.get("out", "/home/tz/workspace/airbot/sim-rerun/artifacts/ds_pick_place")
    max_updates = int(col.get("max_updates", 400))
    fps = int(col.get("fps", 30))          # basis.yaml 里 update_freq: 30

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(out):
        shutil.rmtree(out)

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,),
                              "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
        "action": {"dtype": "float32", "shape": (8,),
                   "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
    }
    for name in CAMS:
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": (IMG_HW[0], IMG_HW[1], 3),
            "names": ["height", "width", "channel"]}

    ds = LeRobotDataset.create(repo_id="local/airbot_sim_pick_place", fps=fps,
                               root=out, features=features, use_videos=True)

    n_ep, n_ok, seed = 0, 0, 1000
    while n_ep < n_ep_target:
        seed += 1
        raw["task"]["seed"] = seed                      # 每条轨迹换一个随机种子 -> 物块位置不同
        task_file = prepare_task_file(OmegaConf.create(raw))
        runner = TaskRunner().from_config(task_file)
        try:
            runner.reset()
            backend = runner._context.backend
            env_n = None
            buf = None
            done_flag = None
            steps = 0
            while steps < max_updates:
                obs = backend.env.capture_observation()
                if env_n is None:
                    env_n = np.asarray(obs[STATE_KEYS[0]]["data"]).shape[0]
                    buf = [[] for _ in range(env_n)]
                for i in range(env_n):
                    frame = {
                        "observation.state": _vec(obs, STATE_KEYS, i),
                        "action": _vec(obs, ACTION_KEYS, i),
                    }
                    for name, key in CAMS.items():
                        img = np.asarray(obs[key]["data"], dtype=np.uint8)[i]
                        frame[f"observation.images.{name}"] = _resize(img, IMG_HW)
                    buf[i].append(frame)
                upd = runner.update()
                steps += 1
                if bool(np.all(upd.done)):
                    done_flag = np.asarray(upd.success).reshape(-1)
                    break
            if done_flag is None:
                done_flag = np.zeros(env_n or 1, dtype=bool)
        finally:
            runner.close()

        # 只保留成功的轨迹 —— 失败的示教数据会毒化策略
        for i in range(env_n):
            if n_ep >= n_ep_target:
                break
            if not bool(done_flag[i]):
                print(f"  [skip] seed={seed} env={i} 失败，丢弃")
                continue
            for fr in buf[i]:
                fr["task"] = TASK_TEXT   # lerobot 0.6+ 把 task 放进 frame 字典，不再是 add_frame 的参数
                ds.add_frame(fr)
            ds.save_episode()
            n_ep += 1; n_ok += 1
            print(f"  [keep] seed={seed} env={i} 共 {len(buf[i])} 帧  -> 累计 {n_ep}/{n_ep_target}")

    meta = {"episodes": n_ep, "fps": fps, "img_hw": list(IMG_HW),
            "state_dim": 8, "action_dim": 8, "cameras": list(CAMS),
            "task": TASK_TEXT, "root": out}
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    with open(os.path.join(out, "collect_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


main()

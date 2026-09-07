#!/usr/bin/env python3
"""复位一次 MuJoCo 场景，把观测字典的全部 key / 形状打出来。
这是做数据集之前的第一步：先亲眼确认"特征契约"长什么样，
而不是照抄别人的 yaml —— 90% 的训练报错都源自这里对不上。"""
import sys, numpy as np, hydra
from omegaconf import DictConfig, OmegaConf
from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import TaskRunner


@hydra.main(config_path=str(get_config_dir()), config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig):
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw.pop("recorder", None)
    runner = TaskRunner().from_config(prepare_task_file(cfg))
    try:
        runner.reset()
        obs = runner._context.backend.env.capture_observation()
        print(f"\n{'='*90}\n观测字典共 {len(obs)} 个 key\n{'='*90}")
        for k in sorted(obs):
            d = obs[k].get("data")
            a = np.asarray(d) if d is not None else None
            shp = "None" if a is None else f"{a.shape} {a.dtype}"
            tag = ""
            if k.endswith("/color/image_raw"): tag = "  ← 图像"
            elif k.startswith("action/"):      tag = "  ← 动作"
            elif "joint_state" in k or "pose" in k: tag = "  ← 状态"
            print(f"  {k:<62} {shp}{tag}")
    finally:
        runner.close()


main()

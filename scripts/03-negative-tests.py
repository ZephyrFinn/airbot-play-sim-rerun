#!/usr/bin/env python3
"""
仿真下的三个"反面用例"——面试里比"跑通了"值钱得多的部分。
1) 超关节限位的目标  -> 规划器应当直接拒绝（不是算半天再失败）
2) 自碰撞姿态的目标  -> SRDF disable_collisions 之外的碰撞对会被拦下
3) 正常但大幅度目标  -> 成功，作为对照组
MoveItErrorCodes 常见值: 1=SUCCESS, -1=FAILURE, -2=PLANNING_FAILED,
                         -4=CONTROL_FAILED, -10=START_STATE_IN_COLLISION,
                         -12=GOAL_IN_COLLISION, -18=NO_IK_SOLUTION,
                         -31=INVALID_GOAL_CONSTRAINTS
"""
import json, sys, time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PlanningOptions

JOINTS = [f"joint{i}" for i in range(1, 7)]
CASES = {
    "1_超限位(joint2=-3.5rad)": [0.0, -3.5, 0.0, 0.0, 0.0, 0.0],
    "2_疑似自碰撞(全折叠)":      [0.0, -2.5, 2.9, 0.0, 1.5, 0.0],
    "3_对照组(正常大幅度)":      [1.2, -0.8, 1.0, 0.5, -0.6, 0.3],
}


def build(target, plan_only=True):
    g = MoveGroup.Goal()
    g.request.group_name = "airbot_play_g2"
    g.request.num_planning_attempts = 5
    g.request.allowed_planning_time = 3.0
    g.request.max_velocity_scaling_factor = 0.1
    g.request.max_acceleration_scaling_factor = 0.1
    c = Constraints()
    for n, p in zip(JOINTS, target):
        jc = JointConstraint()
        jc.joint_name, jc.position = n, float(p)
        jc.tolerance_above = jc.tolerance_below = 1e-3
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    g.request.goal_constraints.append(c)
    o = PlanningOptions(); o.plan_only = plan_only
    g.planning_options = o
    return g


rclpy.init()
n = Node("negative_tests")
cli = ActionClient(n, MoveGroup, "/move_action")
cli.wait_for_server()
out = {}
for name, tgt in CASES.items():
    t0 = time.time()
    f = cli.send_goal_async(build(tgt))
    rclpy.spin_until_future_complete(n, f)
    gh = f.result()
    if not gh.accepted:
        out[name] = {"accepted": False}
        continue
    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(n, rf)
    r = rf.result().result
    out[name] = {
        "target": tgt,
        "error_code": r.error_code.val,
        "traj_points": len(r.planned_trajectory.joint_trajectory.points),
        "wall_time_s": round(time.time() - t0, 3),
    }
print(json.dumps(out, indent=2, ensure_ascii=False))
n.destroy_node(); rclpy.shutdown()

#!/usr/bin/env python3
"""补一个真正触发碰撞检测的用例。
之前 03-negative-tests.py 里挑的"折叠姿态"其实没碰撞（SRDF 里相邻连杆的碰撞对被 disable 了），
所以规划成功了。这里改用 PlanningScene 往工作空间正中间塞一个障碍物盒子，
让目标位置被占住，看 MoveIt 的反应。

这比"随便挑个姿态碰运气"严谨得多，也是面试里"你怎么验证碰撞检测真的在工作"的答案。
"""
import json, time, sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PlanningOptions, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

J = [f"joint{i}" for i in range(1, 7)]
TARGET = [0.5, -0.3, 0.4, 0.0, 0.2, 0.0]     # 和 02 脚本同一个已知可达的目标


def build_goal(target, plan_only=True):
    g = MoveGroup.Goal()
    g.request.group_name = "airbot_play_g2"
    g.request.num_planning_attempts = 5
    g.request.allowed_planning_time = 3.0
    g.request.max_velocity_scaling_factor = 0.1
    g.request.max_acceleration_scaling_factor = 0.1
    c = Constraints()
    for n_, p in zip(J, target):
        jc = JointConstraint(); jc.joint_name = n_; jc.position = float(p)
        jc.tolerance_above = jc.tolerance_below = 1e-3; jc.weight = 1.0
        c.joint_constraints.append(jc)
    g.request.goal_constraints.append(c)
    o = PlanningOptions(); o.plan_only = plan_only; g.planning_options = o
    return g


def run(node, cli, target):
    t0 = time.time()
    f = cli.send_goal_async(build_goal(target))
    rclpy.spin_until_future_complete(node, f)
    gh = f.result()
    if not gh.accepted:
        return {"accepted": False}
    rf = gh.get_result_async(); rclpy.spin_until_future_complete(node, rf)
    r = rf.result().result
    return {"error_code": r.error_code.val,
            "traj_points": len(r.planned_trajectory.joint_trajectory.points),
            "wall_s": round(time.time() - t0, 3)}


rclpy.init()
n = Node("collision_test")
cli = ActionClient(n, MoveGroup, "/move_action"); cli.wait_for_server()
pub = n.create_publisher(PlanningScene, "/planning_scene", 10)
out = {}

out["1_无障碍物"] = run(n, cli, TARGET)

# 往场景里加一个 0.4×0.4×0.4 的盒子，压在机械臂前方
co = CollisionObject()
co.header.frame_id = "base_link"
co.id = "big_box"
prim = SolidPrimitive(); prim.type = SolidPrimitive.BOX; prim.dimensions = [0.4, 0.4, 0.4]
pose = Pose(); pose.position.x = 0.30; pose.position.y = 0.0; pose.position.z = 0.30
pose.orientation.w = 1.0
co.primitives.append(prim); co.primitive_poses.append(pose)
co.operation = CollisionObject.ADD

scene = PlanningScene(); scene.is_diff = True
scene.world.collision_objects.append(co)
for _ in range(6):
    pub.publish(scene); rclpy.spin_once(n, timeout_sec=0.2)
time.sleep(1.5)

out["2_加了障碍物盒子"] = run(n, cli, TARGET)

# 清掉障碍物，确认能恢复
co.operation = CollisionObject.REMOVE
scene2 = PlanningScene(); scene2.is_diff = True
scene2.world.collision_objects.append(co)
for _ in range(6):
    pub.publish(scene2); rclpy.spin_once(n, timeout_sec=0.2)
time.sleep(1.5)
out["3_移除障碍物后"] = run(n, cli, TARGET)

out["说明"] = {
    "障碍物": "base_link 系下 0.4^3 m 盒子，中心 (0.30, 0, 0.30)",
    "判读": "加障碍后 error_code 应变成负数（-12 GOAL_IN_COLLISION 或 -2/99999 规划失败）；移除后应恢复 1",
}
print(json.dumps(out, indent=2, ensure_ascii=False))
n.destroy_node(); rclpy.shutdown()

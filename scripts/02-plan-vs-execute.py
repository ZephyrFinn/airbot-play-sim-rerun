#!/usr/bin/env python3
"""
用 MoveGroup Action 复现 SOP 里 RViz 上的两个按钮：
    Plan            -> planning_options.plan_only = True
    Plan & Execute  -> planning_options.plan_only = False

关键点（面试常问）：
  两者都会在 RViz 里播放同一段轨迹动画，所以肉眼分不出来。
  唯一可靠的判据是 /joint_states 里的 position 有没有真的变。
    Plan          : 只算轨迹并预览，不下发 -> position 不变
    Plan & Execute: 把轨迹交给 airbot_play_g2_controller 的
                    follow_joint_trajectory Action -> position 变到目标
"""
import sys, json, time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PlanningOptions

GROUP = "airbot_play_g2"
JOINTS = [f"joint{i}" for i in range(1, 7)]


class PlanExec(Node):
    def __init__(self):
        super().__init__("plan_vs_execute")
        self.last = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.cli = ActionClient(self, MoveGroup, "/move_action")

    def _cb(self, msg):
        self.last = dict(zip(msg.name, msg.position))

    def snapshot(self, timeout=5.0):
        """取一帧当前关节角，等价于 SOP 里的 ros2 topic echo --once /joint_states"""
        self.last = None
        t0 = time.time()
        while self.last is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return [round(self.last[j], 6) for j in JOINTS] if self.last else None

    def move(self, target, plan_only):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.1      # 仿真也用低速，习惯要和真机一致
        req.max_acceleration_scaling_factor = 0.1
        c = Constraints()
        for name, pos in zip(JOINTS, target):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        opt = PlanningOptions()
        opt.plan_only = plan_only          # <<< 这一个布尔量就是两个按钮的全部差别
        goal.planning_options = opt

        self.cli.wait_for_server()
        fut = self.cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        assert gh.accepted, "goal 被拒绝"
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        res = rf.result().result
        return {
            "error_code": res.error_code.val,        # 1 = SUCCESS
            "planning_time_s": round(res.planning_time, 4),
            "traj_points": len(res.planned_trajectory.joint_trajectory.points),
        }


def main():
    rclpy.init()
    n = PlanExec()
    target = [0.5, -0.3, 0.4, 0.0, 0.2, 0.0]
    out = {"group": GROUP, "target": target}

    out["before"] = n.snapshot()
    out["plan_only_result"] = n.move(target, plan_only=True)
    time.sleep(1.0)
    out["after_plan"] = n.snapshot()

    out["plan_execute_result"] = n.move(target, plan_only=False)
    time.sleep(1.0)
    out["after_execute"] = n.snapshot()

    d = lambda a, b: [round(y - x, 6) for x, y in zip(a, b)]
    out["delta_plan"] = d(out["before"], out["after_plan"])
    out["delta_execute"] = d(out["after_plan"], out["after_execute"])
    out["verdict"] = {
        "plan 没有改变关节": all(abs(v) < 1e-4 for v in out["delta_plan"]),
        "execute 改变了关节": any(abs(v) > 1e-3 for v in out["delta_execute"]),
        "execute 到位误差": [round(abs(a - b), 6) for a, b in zip(out["after_execute"], target)],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    n.destroy_node(); rclpy.shutdown()
    return 0 if all(out["verdict"][k] for k in ("plan 没有改变关节", "execute 改变了关节")) else 1


sys.exit(main())

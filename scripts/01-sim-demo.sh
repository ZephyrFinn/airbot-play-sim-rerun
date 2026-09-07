#!/usr/bin/env bash
# 仿真（MoveIt2 mock_components/GenericSystem 假硬件）Demo
# 起来的东西：move_group + ros2_control(controller_manager) + joint_state_broadcaster
#             + airbot_play_g2_controller + robot_state_publisher + RViz2
set -e
. /home/tz/workspace/airbot/sim-rerun/scripts/00-shell.sh
exec ros2 launch airbot_play_movit_config demo.launch.py "$@"

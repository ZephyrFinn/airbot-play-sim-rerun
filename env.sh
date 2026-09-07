# 本次仿真复现的统一环境变量（每个新终端 source 一次）
export SIMRUN=/home/tz/workspace/airbot/sim-rerun
export WS=$SIMRUN/ws/AIRBOT-Play-Hardware-with-Moveit2
export REPO=$WS
export ROS_DOMAIN_ID=42          # 与机器上其它 ROS 进程隔离，避免话题串台
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

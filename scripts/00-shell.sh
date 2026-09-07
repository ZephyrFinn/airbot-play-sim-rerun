# 每个新终端都要跑这一段（用 source/. 加载，不要直接执行）
#   1) 本次复现的路径变量  2) 系统 ROS 2 Jazzy  3) 本工作空间的 airbot_* 包
# 关键：ROS 2 的 setup 脚本按 shell 分家。zsh 必须 setup.zsh，bash 必须 setup.bash，
#       混用会报 "没有那个文件或目录: .../setup.sh"（它会用错误的相对路径去找 setup.sh）
. /home/tz/workspace/airbot/sim-rerun/env.sh
if [ -n "$ZSH_VERSION" ]; then _sfx=zsh; else _sfx=bash; fi
source /opt/ros/jazzy/setup.$_sfx
source "$WS/install/setup.$_sfx"
unset _sfx

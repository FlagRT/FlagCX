#!/bin/bash
# FlagCX / HCCL 多进程通信必需环境变量（Ascend 910C · CANN 9.0.0 · driver 25.5.0）
#
# 背景：CANN 9.0 的 HCCL 多进程通信（HcclCommInitRootInfo）未设置
# HCCL_NPU_SOCKET_PORT_RANGE 时，NPU socket 走默认单端口 16666 并尝试绑定
# 设备 HCCN 高速网口地址（192.26.2.199 等）——本机从未配置 HCCN 接口，
# HCCL 接口校验失败即报错误码 7（HCCL_E_UNAVAIL）。
# 详见 workspace/FlagCX通信接入-阶段3执行记录.md 附录 A。
#
# 安装（容器内，已执行）：
#   cp scripts/setup_hccl_env.sh /etc/profile.d/flagcx-hccl.sh
#   echo 'source /etc/profile.d/flagcx-hccl.sh' >> /root/.bashrc
# 效果：login shell（bash -l）与交互 shell（docker exec -it）自动生效；
# 非交互 exec（bash -c）需手动 source 或显式 export。

export HCCL_NPU_SOCKET_PORT_RANGE=16666,16676

# 16 rank 全量测试时若报端口不足，放宽为（任选其一）：
#   export HCCL_NPU_SOCKET_PORT_RANGE=16666,16766

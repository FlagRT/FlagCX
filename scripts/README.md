# scripts/ — Ascend 本地诊断与环境工具（非上游代码）

本目录为 FlagOS 运行时层在 Ascend 910C（CANN 9.0.0 / driver 25.5.0）上
打通 FlagCX 的本地工具，**不属于上游 FlagCX 仓库内容**，提交时保持独立 commit。

| 文件 | 用途 |
|---|---|
| setup_hccl_env.sh | HCCL 多进程必需环境变量（HCCL_NPU_SOCKET_PORT_RANGE）。安装：拷到 /etc/profile.d/ + .bashrc source（详见文件头注释） |
| bare_hccl_test.cpp | 裸 HCCL 最小复现（不经 FlagCX，直接 HcclCommInitRootInfo + AllReduce），排障专用 |

## bare_hccl_test 编译与运行（容器内）

```bash
mpicxx -o /tmp/bare_hccl_test scripts/bare_hccl_test.cpp -lmpi_cxx \
  -I/usr/local/Ascend/ascend-toolkit/latest/include \
  -L/usr/local/Ascend/ascend-toolkit/latest/lib64 -lascendcl -lhccl

source scripts/setup_hccl_env.sh      # 或依赖 login/交互 shell 自动生效
cd /tmp && mpirun -np 2 --allow-run-as-root ./bare_hccl_test
# 期望：HcclCommInitRootInfo -> 0、HcclAllReduce -> 0、result=3.0 OK、=== PASSED ===
```

## 排障要点速记

- 日志开关：`ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_SLOG_PRINT_TO_STDOUT=1`
  （CANN 9.0 无 HCCL_DEBUG_LEVEL 变量）；日志落 /root/ascend/log/plog/
- 容器网络必须 host（bridge 看不到 HCCN 地址）；驱动管理通道并发容器上限 ≈3
- 完整失败经验见 workspace/FlagCX通信接入-阶段3执行记录.md §6

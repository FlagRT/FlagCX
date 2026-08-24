# 昇腾 Ascend A 线验证报告（2026-08-24）

> 分支：`kistich/ascend-dev1.0`（基于 dev-1.0 @ 4e0e0cb）
> 环境：910C（4 NPU / 8 chip, HBM 64GB, CANN 9.0.0, 驱动 25.5.0）+
> `flagrt/ascend-operator-runtime:0.1.0-cann9.0-py311-torch2.10-arm64` 官方镜像 +
> torch_npu 2.10.0
> 定位：把 910C 双卡 HCCL 实测的四层根因修复带到 dev-1.0 基线（A 线主线，
> torch_npu 路线）；原 `kistich/ascend-flagcx-adapt`（torch_fl/B 线基线）降级为
> 预研支线，不再作为开发主线。

## 一、分支内容（3 个 commit）

| commit | 内容 |
|---|---|
| 5f7ad78 | 四层根因修复 rebase 到 dev-1.0：HcclRootInfo(4108B)>flagcxUniqueId(256B) 溢出 → thread_local + bootstrap 广播完整 RootInfo；设备绑定注释与处理；**保留** dev-1.0 已合入的 broadcast 字节数修复（c0494fe） |
| 60e0695 | 移除对 torch_fl `GetCurrentStream` 的硬链接依赖（A 线无 torch_fl/libflagos.so，`undefined symbol`）；stream 0 统一走 `GetFlagcxCurrentAclStream()` 的 dlsym 动态解析 + 自建流 fallback |
| 0686805 | **A 线 stream 语义修复**：`getStreamByIndex(0)` 在 ASCEND 分支优先解析 torch_npu 当前流——`c10::impl::getDeviceGuardImpl(PrivateUse1)->getStream()` 取 c10::Stream，再 `dlopen("libtorch_npu.so", RTLD_NOLOAD)` + `dlsym("_ZNK7c10_npu9NPUStream6streamEv")` 转为 aclrtStream（NPUStream 内存布局以 c10::Stream 开头，this 指针可直接复用）。**注意：torch_npu 对默认流返回 nullptr（ACL null=默认流语义），nullptr 是有效流**，不得 fallback 到自建流，否则与 tensor op 无 happens-before → allgather 数据全 0。不链接 libtorch_npu.so，保持 torch 插件与 host runtime 解耦（延续 8ebdeba 的设计） |

## 二、已验证通过项（torch_npu + `backend="flagcx"`）

- **双卡 allgather 数据正确性**：两 rank 两轮 `out=[1,2]` / `out2=[10,11]`
  全对，多轮稳定复现；修复前的 `free(): invalid pointer` 退出崩溃随 stream
  修复一并消失（测试：runtime-team `dev/device-context/benchmarks/test_ag_npu.py`）
- **allreduce 压测**：1000 次 × 200MB bf16，中位 2.1ms（首帧 1.74s 为通信建立，正常）
- **小模型 DDP**：50M 参数 MLP × 3000 iter，稳定 3.2ms/iter 无退化
- **通信链路**：bootstrap 广播 RootInfo、HCCS 同节点通信、E_PARA/网卡根因均未复现

## 三、遗留问题（已知缺陷，待下一阶段）

**现象**：Qwen2.5-1.5B（1.54B params, bf16）+ DDP + flagcx backend 训练，
**step ≈ 765 起**每步耗时 0.1s → ~2.4s，rank0 loss 在 1.0000 / 2.0954 间
精确交替（模型参数停止更新的表象），吞吐从 5032 tok/s 单调下滑。原生
`backend="hccl"`（torch_npu ProcessGroupHCCL）同脚本训练 2481 步全程健康：
loss 收敛 1.9472、吞吐稳定 5428 tok/s —— 证明环境/模型/数据无问题，问题
定位在 **flagcx backend 承载大模型 DDP 负载的路径**。

**已排除**：allgather / allreduce 单原语、小模型 DDP、HBM（23GB/64GB 非 OOM）、
transformers 版本（与 B 线一致 5.15.1）。

**疑点方向**：DDP bucket 重建（rebuild_buckets）时的 stream 切换交互；
长时运行下 `thread_local` stream 缓存与 reducer 内部流的错位；特定
collective 序列（大 tensor 分桶 + 计算交织）。

**诊断资产**（runtime-team `dev/device-context/benchmarks/`）：
`test_ag_npu.py`（allgather）、`train_qwen_1_5b_npu.py`（训练，hccl/flagcx
双后端可切换，一行 sed 复现分界实验）。

## 四、路线状态说明

- **A 线（主线）**：厂商插件（torch_npu）+ FlagGems + FlagCX —— 本分支即 A 线
  基线；torch_npu 原生 hccl 训练闭环已达成（见上），FlagCX 适配以本分支推进。
- **B 线（预研支线）**：torch_fl 路线降级，原分支 `kistich/ascend-flagcx-adapt`
  保留不删、不再承担交付；其中 stream 语义等结论已吸收进本分支。

# FlagcxCollective —— 昇腾持久 kernel（DAG 引擎设备侧消费方）

> O2 阶段三 #7（方案 A1：独立 msOpGen 工程）。对标 CUDA `flagcxCollectiveKernel`
> （`flagcx/adaptor/kernel/nvidia/flagcx_reduce_kernel_device.cu`）。
> 设计细节见 `flagos-demos/docs/O2_phase3_impl.md` §2、§0。

## 作用

host FIFO（`flagcx_reduce_kernel_host.cc` enqueue）→ 持久 kernel 消费（本 op）→ reduce
（`out = fst + snd`）→ `setComplete`。对应 NCCL GIN Proxy 的 lock-free GPU-to-CPU queues：
host 只写 FIFO，device kernel 轮询消费。

## FIFO 布局（与 flagcx_kernel_core.h:44-60 对齐，单位 uint64）

```
[0]=capacity [1]=consumed [2]=produced [3]=terminate [4..]=trigger[]
trigger（flagcxReduceTrigger::value[4]，alignas(16)，32B/个）：
  [0]=fst [1]=snd [2]=out
  [3]=count(32)|nthreads(16)|datatype(4)|redop(4)|state(2)|reserved(1)
state: 0=Available 1=Enqueued 2=Inprogress 3=Complete
```

## 语义映射（CUDA → Ascend C，关键差异）

| CUDA | Ascend C | 依据 |
|---|---|---|
| `Atomic::load` 轮询控制槽 | volatile `__gm__` 读 + **每轮 `DataCacheCleanAndInvalid`(dcci) 失效 L2** | 探针 T5c（host→kernel 可见性必需） |
| 无限循环等 terminate | **SPIN_MAX=1e11 上限**，超时写 `0xDEADBEEF` 退出 | 探针 T5a 教训（死循环 kernel 残留污染调度） |
| `setComplete` = Atomic fetchOr | 写 state=Complete + **dcci（Clean 语义强制写回主存）** | 🔴 新发现对策（host pollState 快速可见） |
| `slot = consumed & (cap-1)` | 同（host 容量 2 的幂，FLAGCX_FIFO_CAPACITY=128） | flagcx_reduce_kernel_device.cu:133 |

第一版限制：单核单块（SetBlockDim(1)），仅 `flagcxFloat32(7)` + `flagcxSum(0)`，
count 级标量循环（正确性优先，性能后续优化）。

## 构建（910C 宿主机 CANN 8.5.0）

```bash
cd flagcx/adaptor/kernel/ascend
# 若从零生成骨架（json 是唯一真源）：
export PATH=/usr/local/Ascend/cann-8.5.0/aarch64-linux/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p /tmp/flagcx_op_ascend && chmod go-w /tmp/flagcx_op_ascend
msopgen gen -i flagcx_collective.json -c ai_core-Ascend910_9382 -lan cpp -out ./FlagcxCollective
# 覆盖 op_kernel/op_host 后编译：
cd FlagcxCollective && source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0
./build.sh
# 产物：build_out/custom_opp_*_aarch64.run
```

## 安装

```bash
cd build_out && ./custom_opp_*_aarch64.run   # 直接执行即安装（--install 是 help）
# 安装到 CANN opp/vendors/customize/，生成 aclnn 头：
#   opp/vendors/customize/op_api/include/aclnn_flagcx_collective.h
```

## 验证（#7.5 单测，见 docs/O2_phase3_impl.md §3）

- 用例 1：host enqueue（模拟 flagcx_reduce_kernel_host.cc 语义）→ kernel 消费 → 断言
  `out[i] == fst[i]+snd[i]`，100 轮 trigger + produced 回绕
- 用例 2：kernel setComplete + dcci clean → host pollState 不 invalidate，断言 <10ms 读到 Complete
- 编译链接（8.5.0 分架构）参考 `scripts/p2_atomic_probe/`：
  `-I aarch64-linux/include -I vendors/customize/op_api/include -L aarch64-linux/lib64
   -L vendors/customize/op_api/lib -lascendcl -lnnopbase -lopapi -lcust_opapi -lacl_op_compiler`

## 接线（#7.2，后续）

`uni_runner_impl.cc:1870` 的 `#ifdef COMPILE_KERNEL_HOST` 分支内，昇腾编译时替换为 aclnn
异步 launch（不 sync，kernel 驻留）：

```cpp
uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
aclnnFlagcxCollectiveGetWorkspaceSize(fifoTensor, &wsSize, &exec);
aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnFlagcxCollective(stream, fifoTensor, ws, wsSize, exec);
// kernel 驻留期间不能 sync 该 stream（terminate 后由 runUniRunner 尾部统一 sync）
```

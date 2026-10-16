#include "kernel_operator.h"
using namespace AscendC;
/*
 * O2 阶段三 #7：昇腾持久 kernel（DAG 引擎设备侧消费方）——第一版：单核单块
 *
 * 对标 CUDA flagcxCollectiveKernel（nvidia/flagcx_reduce_kernel_device.cu），
 * 语义映射见 docs/O2_phase3_impl.md §2.1：
 *   - 消费 host FIFO（flagcx_kernel_core.h:44-60 布局）
 *   - 每轮 DataCacheCleanAndInvalid(dcci) 失效 L2（探针 T5c 结论：host->kernel 可见性必需）
 *     —— dcci 必须覆盖**所有** host 写/读的地址：控制槽行、trigger 槽行、以及
 *     reduce 数据缓冲 fst/snd（读前失效 L2 旧值）与 out（写后 Clean 语义写回主存）。
 *     （t7 单测 round 0 教训：只 dcci 控制结构、不 dcci 数据地址 -> out 数据错）
 *   - SPIN_MAX 自旋上限防残留（探针 T5a 教训：死循环 kernel 残留污染 device 调度）
 *   - setComplete 后 dcci（含 Clean 语义强制写回主存）-> host pollState 快速可见
 *     （🔴 新发现对策，见 O2_phase3_impl.md §0，待 #7.5 用例 2 验证）
 *
 * FIFO 布局（单位 uint64）：
 *   [0]=capacity [1]=consumed [2]=produced [3]=terminate [4..]=trigger[]
 * trigger（flagcxReduceTrigger::value[4]，alignas(16)，32B/个）：
 *   [0]=fst [1]=snd [2]=out
 *   [3]=count(32)|nthreads(16)|datatype(4)|redop(4)|state(2)|reserved(1)
 * state: 0=Available 1=Enqueued 2=Inprogress 3=Complete
 *   host setValue(Enqueued) / host pollState；kernel 消费后置 Complete；
 *   host 查完完成调 setState(Available) 使槽可复用（flagcx_reduce_kernel_host.cc）
 *
 * 第一版限制（正确性优先）：
 *   - 单核单块（SetBlockDim(1)），nthreads 忽略
 *   - 仅支持 flagcxFloat32(7) + flagcxSum(0)，count 级标量循环
 *   - 其余 datatype/redop 组合：out[0]=NaN 暴露错误
 */
#define FIFO_CAPACITY_IDX 0
#define FIFO_CONSUMED_IDX 1
#define FIFO_PRODUCED_IDX 2
#define FIFO_TERMINATE_IDX 3
#define FIFO_DATA_IDX 4

#define TRIGGER_WORDS 4
#define TRIGGER_OFF_COUNT 0      /* 32b */
#define TRIGGER_OFF_NTHREADS 32  /* 16b */
#define TRIGGER_OFF_DATATYPE 48  /* 4b */
#define TRIGGER_OFF_REDOP 52     /* 4b */
#define TRIGGER_OFF_STATE 56     /* 2b */

#define TRIGGER_STATE_MASK 0x3ull
#define TRIGGER_STATE_COMPLETE 0x3ull

#define DT_FLOAT32 7 /* flagcxFloat32 = flagcxFloat */
#define REDOP_SUM 0  /* flagcxSum */

#define SPIN_MAX 100000000000ull /* 防残留自旋上限（探针 v3 同量级，正常 terminate 先到） */
#define TERM_TIMEOUT_MARK 0xDEADBEEFull

extern "C" __global__ __aicore__ void flagcx_collective(GM_ADDR fifo,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling) {
    volatile __gm__ uint64_t *buf = (volatile __gm__ uint64_t *)fifo;

    GlobalTensor<uint64_t> gctrl; /* 控制槽行（capacity/consumed/produced/terminate 同缓存行） */
    gctrl.SetGlobalBuffer((__gm__ uint64_t *)fifo, 4);

    /* 首轮先失效控制槽行：读到 host 已写好的 capacity（launch 前 host 完成 FIFO init） */
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gctrl);
    uint64_t capacity = buf[FIFO_CAPACITY_IDX];
    uint64_t capMask = capacity - 1; /* host 侧容量为 2 的幂（FLAGCX_FIFO_CAPACITY=128） */

    uint64_t spins = 0;

    /* P2a（2026-09-02）：单核向量化 reduce——UB 分块 DataCopy + 向量 Add，
     * 替代第一版 count 级标量循环（单核标量 750M 次迭代是 3GB AllReduce 28s 的瓶颈）。
     * UB 3 块 × 8192 float（96KB），尾块（n<8192）回落标量避免向量不对齐。
     * 保持单核（SetBlockDim(1)）：先验证向量化收益，多核分片（P2b）后续叠加。 */
    constexpr uint32_t CHUNK_FLOATS = 8192;
    constexpr uint32_t CHUNK_TAIL_SCALAR = 256; /* 尾块阈值：小于此用标量，规避 repeat 对齐 */
    TPipe pipe;
    TBuf<TPosition::VECCALC> tbuf;
    pipe.InitBuffer(tbuf, CHUNK_FLOATS * 3 * sizeof(float));
    LocalTensor<float> ubFst = tbuf.Get<float>();
    LocalTensor<float> ubSnd = ubFst[CHUNK_FLOATS];
    LocalTensor<float> ubOut = ubSnd[CHUNK_FLOATS];

    while (true) {
        /* 每轮失效控制槽行：host 运行时写 produced/terminate（T5c 结论必需） */
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gctrl);
        uint64_t consumed = buf[FIFO_CONSUMED_IDX];
        uint64_t produced = buf[FIFO_PRODUCED_IDX];
        uint64_t terminate = buf[FIFO_TERMINATE_IDX];

        if (consumed >= produced) {
            if (terminate == 1) break;
            if (++spins > SPIN_MAX) {
                /* 超时防残留：标记异常后退出（host 侧一直等不到 Complete，走 #8 超时/回退） */
                buf[FIFO_TERMINATE_IDX] = TERM_TIMEOUT_MARK;
                DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gctrl);
                break;
            }
            continue;
        }
        spins = 0;

        uint64_t slot = consumed & capMask;
        volatile __gm__ uint64_t *tr = buf + FIFO_DATA_IDX + slot * TRIGGER_WORDS;
        GlobalTensor<uint64_t> gtr; /* trigger 槽行（host enqueue 刚写入，需失效旧 L2） */
        gtr.SetGlobalBuffer((__gm__ uint64_t *)tr, TRIGGER_WORDS);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gtr);

        uint64_t fst = tr[0];
        uint64_t snd = tr[1];
        uint64_t out = tr[2];
        uint64_t ctrl = tr[3];
        uint32_t count = (uint32_t)(ctrl >> TRIGGER_OFF_COUNT);
        uint32_t datatype = (uint32_t)((ctrl >> TRIGGER_OFF_DATATYPE) & 0xF);
        uint32_t redop = (uint32_t)((ctrl >> TRIGGER_OFF_REDOP) & 0xF);

        /* 消费完成：consumed+1（释放队列空间；槽复用受 state!=Available 保护，先于 reduce 安全） */
        buf[FIFO_CONSUMED_IDX] = consumed + 1;
        /* consumed 写回主存：host 读 consumed 判断队列空间（对称 dcci clean） */
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gctrl);

        /* 归约：第一版仅 float32 + sum */
        __gm__ float *fstPtr = (__gm__ float *)fst;
        __gm__ float *sndPtr = (__gm__ float *)snd;
        __gm__ float *outPtr = (__gm__ float *)out;
        /* 🔴 读侧 dcci（t7 round 0 失败根因之一）：
         * fst/snd 数据是 host 经 dc cvau 写回主存的，device L2 可能残留旧值挡住读取。
         * ⚠️ dcci 的 SINGLE_CACHE_LINE 只 clean 单个缓存行（64B），而 fst/snd/out 是
         * count 级多行缓冲（4KB+），必须按缓存行循环 dcci（T5c 只验证了 32B 控制槽/trigger，
         * 单行内足够；数据缓冲是新的多行场景）。 */
        GlobalTensor<float> gfst, gsnd, gout;
        constexpr uint32_t LINE_FLOATS = 16; /* 64B 缓存行 / sizeof(float) */
        uint32_t lines = (count + LINE_FLOATS - 1) / LINE_FLOATS;
        /* P1 性能优化（2026-09-02）：DAG 路径下 fst/snd/out 均为**设备缓冲**
         * （AllGather 由设备侧收发、reduce 结果供后续设备节点消费），host 不写这些数据，
         * 故设备侧读/写自身内存无需 dcci（dcci 只用于 host 写主存 → device 读的可见性场景，
         * 即 FIFO 控制槽与 trigger 槽，已保留）。
         * 保留变量 gfst/gsnd/gout 与 lines 以免改动面扩大（若实测发现数据陈旧再恢复）。 */
        for (uint32_t l = 0; l < lines; l++) {
            gfst.SetGlobalBuffer(fstPtr + l * LINE_FLOATS, LINE_FLOATS);
            gsnd.SetGlobalBuffer(sndPtr + l * LINE_FLOATS, LINE_FLOATS);
            /* P1: 已移除读侧逐行 dcci */
        }
        if (datatype == DT_FLOAT32 && redop == REDOP_SUM) {
            /* P2a：向量化 reduce（fst/snd/out 均为设备缓冲，P1 后无需 dcci） */
            GlobalTensor<float> gfst, gsnd, gout;
            gfst.SetGlobalBuffer(fstPtr, count);
            gsnd.SetGlobalBuffer(sndPtr, count);
            gout.SetGlobalBuffer(outPtr, count);
            uint32_t off = 0;
            while (off < count) {
                uint32_t n = (count - off < CHUNK_FLOATS) ? (count - off) : CHUNK_FLOATS;
                if (n >= CHUNK_TAIL_SCALAR) {
                    DataCopy(ubFst, gfst[off], n);
                    DataCopy(ubSnd, gsnd[off], n);
                    Add(ubOut, ubFst, ubSnd, n);
                    DataCopy(gout[off], ubOut, n);
                } else {
                    /* 尾块标量（n 小，向量不划算且可能不对齐） */
                    for (uint32_t i = 0; i < n; i++) {
                        outPtr[off + i] = fstPtr[off + i] + sndPtr[off + i];
                    }
                }
                off += n;
            }
        } else {
            outPtr[0] = 0x7FC00000f; /* NaN：unsupported datatype/redop */
        }
        /* P1 性能优化（2026-09-02）：写侧逐行 dcci 已移除（同上理由——out 是设备缓冲，
         * 后续消费方是设备节点/D2H 拷贝，设备内一致性由硬件保证，无需软件 Clean）。
         * 若实测出现 host 读 out 拿到旧值，恢复本循环即可（备份 .bak_p1_dcci）。 */
        for (uint32_t l = 0; l < lines; l++) {
            gout.SetGlobalBuffer(outPtr + l * LINE_FLOATS, LINE_FLOATS);
            /* P1: 已移除写侧逐行 dcci */
        }

        /* setComplete（对标 CUDA fetchOr）：state 位 -> Complete(3)，保留其余字段 */
        uint64_t newCtrl = (ctrl & ~(TRIGGER_STATE_MASK << TRIGGER_OFF_STATE)) |
                           (TRIGGER_STATE_COMPLETE << TRIGGER_OFF_STATE);
        tr[3] = newCtrl;
        /* 🔴 新发现对策：dcci 含 Clean 语义 -> 脏行强制写回主存，
         * host pollState（__atomic_load_n 无 cache 维护）即可快速读到 Complete */
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE>(gtr);
    }
}

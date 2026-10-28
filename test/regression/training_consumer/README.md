# Training consumer regression

_FlagCX c10d completion, reusable buffers and native DDP process-group lifecycle._

## Scope

This suite adds consumer regressions on top of the Ascend fixes merged in
`62fe6153852658f17e729a06be4da1c766201ef7` (device-context branch,
`kistich/ascend-flagcx-adapt`). It does not reattribute those fixes or replace
the existing collective API suite. The new coverage is repeated buffer reuse
on the caller's current stream and group teardown/reinitialization in a single process.

| Dimension | Values |
| --- | --- |
| Ranks | 2, one local device each |
| AllReduce dtype | FP32, BF16, INT32 |
| Tensor elements | 1, 257, 65,537 |
| Caller stream | default, nondefault |
| Completion | alternating synchronous / `async_op=True` + `Work.wait()` |
| Reuse | 50 calls per configuration, changing data on every call |
| List AllGather | 50 calls per stream, persistent output buffers |
| Group lifecycle | 3 independent init/destroy cycles within each worker |
| Native DDP | 10 SGD steps per cycle, finite loss, actual parameter update |

The first queued consumer clones each result before any host read. A late
write cannot be hidden by synchronizing before that first consumer. DDP
parameters are compared by independent host SHA256s; no additional collective
is used to assert rank agreement.

Each cycle uses its own `PrefixStore` namespace. After destroying a group,
both workers rendezvous through the independent host `TCPStore` before the
next initialization. This follows the required external coordination in
[PyTorch's reinitialization guidance](https://docs.pytorch.org/docs/stable/distributed.html#reinitialization).
It is a bounded healthy-lifecycle test, not general fault recovery support.
Records are saved before native DDP startup and teardown, and include worker
and launcher source SHA256s, so a native abort cannot discard all peer evidence.

## Prerequisites and single command

Use a verified Route A Ascend environment with two exclusively available
devices, PyTorch/torch_npu 2.10, and the installed FlagCX under test. Source
the CANN environment before execution. No runtime-team package, model download,
extra Python dependency or test-specific backend implementation is needed.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python3 test/regression/training_consumer/run.py \
  --backend flagcx --output /absolute/new/results --cycles 3 --iters 50
```

For the CPU control use `--backend gloo`; this omits nondefault NPU streams.
Output directories must be new. Defaults impose a 600-second launcher timeout;
only the launched process group is terminated on timeout. No automatic retries.

Exit codes and `overall:`: PASS/0 requires both ranks, complete checks, all
normal group teardowns, matching DDP hashes and launcher exit 0; FAIL/1 records
a failed result; ERROR/2 identifies missing execution evidence. A pending
teardown record or nonzero launcher cannot be accepted from body results alone.

## Evidence

On npu1-11, driver 26.1.1, logical devices 10/11, public training v3 image
`e13a15d0d02b...` with a mounted `62fe615` override:

| Run | Result | Checks per rank | Group cycles | Native DDP steps per rank |
| --- | --- | ---: | ---: | ---: |
| FlagCX/NPU (three independent launches) | 3/3 PASS | 3,033 per launch | 3 per launch, all closed | 30 per launch |
| Gloo/Linux CPU control | PASS | 1,533 | 3, all closed | 30 |

The 3,033 count comprises 2,700 AllReduce checks, 300 list AllGather checks,
30 finite-loss checks and 3 parameter-update checks. Repeated checks and the
two ranks are not independent performance samples. No latency claim is made.

Machine-readable evidence: [NPU summary](evidence/20261028/npu-summary.json),
[rank 0](evidence/20261028/npu-rank0.json),
[rank 1](evidence/20261028/npu-rank1.json),
[CPU control](evidence/20261028/cpu-summary.json).
Independent repeats: [second](evidence/20261028/npu-repeat2.json),
[third](evidence/20261028/npu-repeat3.json). Total healthy group lifecycles: 9.
The summary records the loaded extension SHA256 and package path.
Full logs remain on `npu1-11` under
`/home/lianzhongyou/lianzhongyou_data/runtime-tests/20261028-communication/evidence/flagcx-coordinated/`
and `.../flagcx-coordinated-cpu/`; they are not committed.

### Negative control and harness correction

The exact same final worker, with one cycle and four iterations, loads the
image-baked `4e0e0cb` package through a per-command `PYTHONPATH` shadow. `ldd`
confirmed that its extension resolves the baked `libflagcx.so` beside it.
The [negative control](evidence/20261028/old-negative.json) returns FAIL/1:
incorrect collective readbacks, native DDP startup error and SIGABRT on exit.
The package path and binary SHA256 differ from the repaired positive arm.

The first old-version run lost one peer record during native abort and was
correctly classified [ERROR](evidence/20261028/old-incomplete.json), not PASS.
Early persistence now retains the observations before DDP startup/teardown.
An [uncoordinated reinitialization run](evidence/20261028/uncoordinated-failure.json)
also failed at cycle 3 bootstrap. The driver now satisfies the documented
external-coordination requirement and isolates each store namespace; three
independent corrected launches pass. These failures remain evidence and are
not attributed to a newly identified FlagCX C++ defect. Healthy lifecycle
coverage does not imply recovery after a failed collective.

Model-level OneRec/DeepFM acceptance and device-prototype consumption live in
[runtime-team's communication direction](https://github.com/FlagRT/runtime-team/tree/dev-1.0/dev/communication).

"""Two-rank stream ordering, buffer reuse and native DDP lifecycle regression.

No runtime-team import. Gloo/CPU is a host control; FlagCX/NPU is acceptance.
The first queued consumer clones each collective result before any host wait.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def digest(model):
    raw = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['flagcx', 'gloo'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--iters', type=int, default=50)
    a = p.parse_args()
    if a.cycles < 1 or a.iters < 1:
        p.error('cycles and iters must be positive')
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        p.error('Exactly two ranks required')
    torch.set_num_threads(1)
    if a.backend == 'flagcx':
        import torch_npu
        import flagcx
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        device = torch.device('npu', int(os.environ['LOCAL_RANK']))
        stream_modes = ['default', 'nondefault']
        library = next(Path(flagcx.__file__).parent.glob('_C*.so'))
        provenance = dict(package=str(Path(flagcx.__file__).resolve()),
                          extension_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    else:
        device, stream_modes, provenance = torch.device('cpu'), ['default'], {}
    result = dict(rank=rank, backend=a.backend, torch=torch.__version__,
                  scope='NPU integration' if a.backend=='flagcx' else 'CPU/Gloo host control',
                  provenance=provenance, worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cycles=[], verdict='ERROR')
    target = a.output/f'rank{rank}.json'
    if target.exists():
        raise FileExistsError(target)
    def save():
        target.write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        # Reinitialization requires coordination outside the destroyed process
        # group. torchrun's agent store also survives group destruction; give
        # each cycle a fresh namespace so stale bootstrap IDs cannot be reused.
        control = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                                is_master=False, timeout=timedelta(seconds=90))
        result['lifecycle_protocol'] = 'unique PrefixStore per cycle; host Store rendezvous after destroy'
        for cycle in range(a.cycles):
            store = dist.PrefixStore(f'training-consumer/cycle-{cycle}/', control)
            dist.init_process_group(a.backend, store=store, rank=rank, world_size=2,
                                    timeout=timedelta(seconds=90))
            record = dict(index=cycle, checks=0, failures=[], ddp_steps=0, teardown='pending')
            result['cycles'].append(record)
            save()
            def check(ok, label):
                record['checks'] += 1
                if not ok and len(record['failures']) < 10:
                    record['failures'].append(label)
            try:
                for mode in stream_modes:
                    stream = torch.npu.Stream() if mode=='nondefault' else None
                    context = torch.npu.stream(stream) if stream is not None else nullcontext()
                    with context:
                        for dtype in (torch.float32, torch.bfloat16, torch.int32):
                            for count in (1, 257, 65537):
                                buffer = torch.empty(count, dtype=dtype, device=device)
                                for step in range(a.iters):
                                    value = (step % 13) + cycle + 1
                                    buffer.fill_(value+rank)
                                    # Exercise both c10d completion paths with one persistent buffer.
                                    async_op = bool(step % 2)
                                    work = dist.all_reduce(buffer, async_op=async_op)
                                    if async_op:
                                        work.wait()
                                    observed = buffer.clone()
                                    expected = torch.full((count,), 2*value+1, dtype=dtype)
                                    check(torch.equal(observed.cpu(), expected),
                                          f'all_reduce/{mode}/{dtype}/{count}/{step}')
                        source = torch.empty(257, dtype=torch.float32, device=device)
                        outputs = [torch.empty_like(source) for _ in range(2)]
                        for step in range(a.iters):
                            source.fill_(step+rank+cycle)
                            dist.all_gather(outputs, source)
                            # Queue consumers before host copies synchronize their outputs.
                            observed = [x.clone() for x in outputs]
                            check(all(torch.equal(x.cpu(), torch.full((257,), float(step+r+cycle)))
                                      for r,x in enumerate(observed)), f'all_gather/{mode}/{step}')
                    save()  # Preserve peer evidence if native DDP startup aborts another rank.
                torch.manual_seed(4321+cycle)
                model = torch.nn.Linear(4, 2).to(device)
                initial = digest(model)
                ddp = DistributedDataParallel(model, **(
                    {'device_ids':[device.index], 'output_device':device.index} if device.type=='npu' else {}))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                for step in range(10):
                    optimizer.zero_grad(set_to_none=True)
                    x = torch.full((3,4), float(rank+1), device=device)
                    loss = ddp(x).square().mean()
                    loss.backward()
                    optimizer.step()
                    check(bool(torch.isfinite(loss).item()), f'ddp/{step}/finite')
                    record['ddp_steps'] += 1
                record['initial_parameter_sha256'] = initial
                record['final_parameter_sha256'] = digest(model)
                check(record['final_parameter_sha256'] != initial, 'ddp/parameter_update')
                del optimizer, ddp, model
            finally:
                save()  # Native cleanup crashes must leave a pending marker.
                dist.destroy_process_group()
                record['teardown'] = 'closed'
                save()
            control.set(f'training-consumer/closed-{cycle}-{rank}', '1')
            control.wait([f'training-consumer/closed-{cycle}-{r}' for r in range(2)],
                         timedelta(seconds=90))
            if record['failures']:
                result['verdict'] = 'FAIL'
                break
        else:
            result['verdict'] = 'PASS'
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        save()
    print(f"rank{rank} overall: {result['verdict']}", flush=True)
    return 0 if result['verdict']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""P800 native DDP vs explicit averaging, using a CPU global-batch oracle.

Run with torchrun --nproc_per_node=2 in the locked Kunlun v2 image.
This is not an Ascend test or a repair; keep failed evidence and launcher exits.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);torch.set_num_threads(1)
    if torch.cuda.device_count()!=2:raise ValueError('Exactly two selected P800 cards required')
    torch.cuda.set_device(rank)
    dist.init_process_group('flagcx',timeout=timedelta(seconds=20))
    report=dict(rank=rank,torch=torch.__version__,checks=[],verdict='ERROR',closed=False)
    target=args.output/f'rank{rank}.json'
    if target.exists():raise FileExistsError(target)
    try:
        for mode in ('explicit','native'):
            torch.manual_seed(923)
            cpu=torch.nn.Linear(4,2)
            model=deepcopy(cpu).cuda(rank)
            wrapped=(DistributedDataParallel(model,device_ids=[rank]) if mode=='native' else model)
            x=torch.arange(16,dtype=torch.float32).reshape(4,4)/16
            cpu(x).square().mean().backward()
            wrapped(x[rank*2:(rank+1)*2].cuda(rank)).square().mean().backward()
            torch.cuda.synchronize()
            if mode=='explicit':
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad);torch.cuda.synchronize();parameter.grad.div_(2)
            torch.cuda.synchronize()
            for (name,p),(_,q) in zip(model.named_parameters(),cpu.named_parameters()):
                observed=p.grad.cpu()
                report['checks'].append(dict(mode=mode,name=name,max_abs=float((observed-q.grad).abs().max()),
                                              passed=bool(torch.allclose(observed,q.grad,atol=2e-5,rtol=2e-4))))
        report['verdict']='PASS' if all(r['passed'] for r in report['checks']) else 'FAIL'
    finally:
        target.write_text(json.dumps(report,indent=2)+'\n')
        dist.destroy_process_group();report['closed']=True
        target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['verdict']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

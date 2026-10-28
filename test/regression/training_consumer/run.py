"""Single entry retaining launcher exits, including native teardown failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['flagcx','gloo'], required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--iters',type=int,default=50)
    p.add_argument('--timeout',type=int,default=600)
    a=p.parse_args()
    if min(a.cycles,a.iters,a.timeout)<1: p.error('positive bounds required')
    a.output.mkdir(parents=True,exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
             '--rdzv_backend=static','--master_addr=127.0.0.1',f'--master_port={port}',
             str(Path(__file__).with_name('worker.py')),'--backend',a.backend,
             '--output',str(a.output),'--cycles',str(a.cycles),'--iters',str(a.iters)]
    timed_out=False
    start=time.time()
    with (a.output/'launcher.log').open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=process.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGTERM)
            try: code=process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); code=process.wait()
    result=dict(argv=command,exit_code=code,timed_out=timed_out,seconds=time.time()-start,
                verdict='ERROR',backend=a.backend,cycles=a.cycles,iters=a.iters,
                launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        ranks=[json.loads((a.output/f'rank{rank}.json').read_text()) for rank in range(2)]
        checks={'launcher_zero':code==0 and not timed_out}
        expected_per_cycle=(2 if a.backend=='flagcx' else 1)*10*a.iters+11
        for rank,r in enumerate(ranks):
            checks[f'rank{rank}_identity']=r['rank']==rank and r['backend']==a.backend
            checks[f'rank{rank}_passed']=r['verdict']=='PASS' and 'error' not in r
            checks[f'rank{rank}_cycles']=len(r['cycles'])==a.cycles
            checks[f'rank{rank}_closed']=all(c['teardown']=='closed' for c in r['cycles'])
            checks[f'rank{rank}_checks']=all(c['checks']==expected_per_cycle and not c['failures']
                                             and c['ddp_steps']==10 for c in r['cycles'])
        checks['same_binary']=ranks[0]['provenance']==ranks[1]['provenance']
        checks['same_worker']=bool(ranks[0].get('worker_sha256')) and ranks[0].get('worker_sha256')==ranks[1].get('worker_sha256')
        checks['ddp_parameters_match']=len(ranks[0]['cycles'])==len(ranks[1]['cycles']) and all(
            x.get('final_parameter_sha256')==y.get('final_parameter_sha256')
            and x.get('initial_parameter_sha256')==y.get('initial_parameter_sha256')
            for x,y in zip(ranks[0]['cycles'],ranks[1]['cycles']))
        incomplete=any(len(r['cycles'])!=a.cycles for r in ranks)
        # A recorded assertion is FAIL; missing execution evidence is ERROR.
        assertion_failure=any(c['failures'] for r in ranks for c in r['cycles'])
        verdict='PASS' if all(checks.values()) else ('ERROR' if incomplete and not assertion_failure else 'FAIL')
        result.update(checks=checks,checks_per_rank=sum(c['checks'] for c in ranks[0]['cycles']),
                      provenance=ranks[0]['provenance'],
                      worker_sha256=ranks[0].get('worker_sha256'),
                      verdict=verdict)
    except (KeyError,FileNotFoundError,ValueError,TypeError) as error:
        result['reason']=str(error)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('overall: '+result['verdict'],flush=True)
    return {'PASS':0,'FAIL':1,'ERROR':2}[result['verdict']]


if __name__=='__main__':
    raise SystemExit(main())

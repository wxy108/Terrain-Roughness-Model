"""Matched complete-gait eager/compiled timing of the repository implementation."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from reproduce_study import MultiScaleEnergy,load_inputs,tensors_for_case,scan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('runs/benchmark.json'))
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('This comparison requires CUDA and a working torch.compile toolchain.')
    if args.repeats<=0:parser.error('--repeats must be positive.')
    torch.set_num_threads(1);records=[]
    height,spacing,origin,centers,_=load_inputs()
    centers_t=torch.as_tensor(centers,dtype=torch.float64,device='cuda')
    with torch.inference_mode():
        for n in [11,19]:
            feet,distance,i,j=tensors_for_case(3.,n,'cuda')
            eager=MultiScaleEnergy(height,spacing,origin,i,j,'cuda')
            compiled=torch.compile(eager,fullgraph=True,dynamic=False)
            batch=4 if n==11 else 12
            compiled(feet[:batch],distance[:batch],centers_t);eager(feet[:1],distance[:1],centers_t)
            torch.cuda.synchronize();times={'eager':[],'compiled':[]}
            for repeat in range(args.repeats):
                results={}
                order=['eager','compiled'] if repeat%2==0 else ['compiled','eager']
                for method in order:
                    torch.cuda.synchronize();t0=time.perf_counter()
                    if method=='eager':
                        # Eager reference returns each kernel to CPU separately,
                        # matching the earlier implementation's transfer pattern.
                        results[method]=np.concatenate([eager(feet[k:k+1],distance[k:k+1],centers_t).cpu().numpy() for k in range(1800)])
                    else:results[method]=scan(compiled,feet,distance,centers_t,batch)
                    torch.cuda.synchronize();times[method].append(time.perf_counter()-t0)
                error=float(abs(np.degrees(np.sqrt(results['eager']))-np.degrees(np.sqrt(results['compiled']))).max())
                assert error<1e-5
                print(json.dumps(dict(segments=n,repeat=repeat+1,eager_s=times['eager'][-1],compiled_s=times['compiled'][-1],difference_deg=error)),flush=True)
            records.append(dict(segments=n,wave=3.,kernels=1800,positions=36300,nonflat_height_scales=9,times=times,
                median_speedup=float(np.median(times['eager'])/np.median(times['compiled'])),max_difference_deg=error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(complete=True,records=records,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        scope='Same session, full scans, CPU output transfer included; warm compiler; alternating order. Inputs already resident on GPU for both methods. This is a repository benchmark, separate from the historical benchmark record.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()

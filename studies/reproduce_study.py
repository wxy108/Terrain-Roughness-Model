"""Recompute the retained 6 x 5 x 10 study without external project paths."""
from pathlib import Path
import argparse,csv,json,time
import numpy as np
import torch
from terrain_roughness.core import _PairEnergy,_kernels,_scan_centers

ROOT=Path(__file__).resolve().parents[1]
SCALES=(.25,.5,.75,1.,1.25,1.5,1.75,2.,2.25)


class MultiScaleEnergy(_PairEnergy):
    """Same pair-angle calculation, sharing height queries across amplitudes."""
    def forward(self,feet,distance,centers):
        slope=self._slopes(feet,distance,centers)
        values=[]
        for scale in SCALES:
            beta=torch.atan(slope*scale)
            values.append(beta.square().sum((1,2),dtype=torch.float64)/(len(self.pair_i)*len(centers)))
        return torch.stack(values,1)


def load_inputs():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    with np.load(ROOT/'data/heightmaps/libo_x1p00.npz') as z:
        height=z['height'].copy();spacing=np.array([float(z['dx_m']),float(z['dy_m'])]);origin=z['origin_xy_m'].copy()
    _,_,_,_,bank=_kernels(3.,19)
    centers=_scan_centers(height.shape,spacing,origin,None,.002,bank)
    assert len(centers)==36300
    return height,spacing,origin,centers,manifest


def verify_scaling(height,manifest):
    with np.load(ROOT/manifest['full_base']['path']) as z:mean=float(z['height'].mean())
    maximum=0.
    for r in manifest['records'][1:]:
        with np.load(ROOT/r['path']) as z:actual=z['height']
        error=float(abs(actual-(mean+r['height_scale']*(height-mean))).max())
        maximum=max(maximum,error)
    assert maximum<1e-12,maximum
    return maximum


def tensors_for_case(wave,segments,device):
    feet,d,i,j,_=_kernels(wave,segments)
    return (torch.as_tensor(feet,dtype=torch.float64,device=device),
            torch.as_tensor(d,dtype=torch.float64,device=device),i,j)


def scan(evaluate,feet,distance,centers,batch=12):
    output=torch.empty((1800,9),dtype=torch.float64,device=feet.device)
    for a in range(0,1800,batch):output[a:a+batch]=evaluate(feet[a:a+batch],distance[a:a+batch],centers)
    return output.cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--compile-gpu',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--waves',type=float,nargs='+',default=[1.5,2.,2.5,3.,3.5,4.])
    parser.add_argument('--segments',type=int,nargs='+',default=[11,13,15,17,19])
    parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/reproduction')
    args=parser.parse_args()
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.compile_gpu and device.type!='cuda':parser.error('--compile-gpu requires CUDA.')
    if args.batch<=0:parser.error('--batch must be positive.')
    torch.set_num_threads(1)
    wall=time.perf_counter();height,spacing,origin,centers,manifest=load_inputs()
    scale_error=verify_scaling(height,manifest)
    with (ROOT/'results/reference_50x36.csv').open() as f:reference=list(csv.DictReader(f))
    expected={(int(r['segments']),float(r['wave']),float(r['height_scale'])):float(r['R_gait_deg']) for r in reference}
    reference_axis={r['height_scale']:r['reference_roughness_deg'] for r in manifest['records']}
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];checks=[];timings=[]
    centers_t=torch.as_tensor(centers,dtype=torch.float64,device=device)
    with torch.inference_mode():
        for n in args.segments:
            # Reuse the compiled shape across all waves at this segment count.
            _,_,i,j,_=_kernels(args.waves[0],n)
            module=MultiScaleEnergy(height,spacing,origin,i,j,device)
            evaluate=torch.compile(module,fullgraph=True,dynamic=False) if args.compile_gpu else module
            for wave in args.waves:
                if (args.output/'STOP_REQUESTED').exists():
                    raise SystemExit('Stopped before next configuration at user request; completed checkpoints retained.')
                feet,distance,_,_=tensors_for_case(wave,n,device)
                if device.type=='cuda':torch.cuda.synchronize()
                t0=time.perf_counter();energy=scan(evaluate,feet,distance,centers_t,args.batch)
                elapsed=time.perf_counter()-t0
                values=np.r_[0.,np.degrees(np.sqrt(energy.mean(0)))]
                checkpoint=args.output/f'N{n}_wave{wave:g}.npz'
                np.savez_compressed(checkpoint,energy=energy)
                for s,value in zip((0.,)+SCALES,values):
                    rows.append(dict(segments=n,wave=wave,height_scale=s,R_reference_deg=reference_axis[s],R_gait_deg=float(value)))
                if args.verify:
                    with np.load(ROOT/f'results/kernel_energy/N{n}_wave{wave:g}.npz') as z:old=z['energy']
                    kernel_error=float(abs(np.degrees(np.sqrt(energy))-np.degrees(np.sqrt(old))).max())
                    final_error=max(abs(value-expected[n,wave,s]) for s,value in zip((0.,)+SCALES,values))
                    assert kernel_error<1e-5 and final_error<1e-5,(kernel_error,final_error)
                    checks.append(dict(segments=n,wave=wave,max_kernel_difference_deg=kernel_error,max_final_difference_deg=float(final_error)))
                timings.append(dict(segments=n,wave=wave,scan_including_first_compile_s=elapsed))
                print(json.dumps(dict(segments=n,wave=wave,wall_s=elapsed,R_x1_deg=float(values[4]))),flush=True)
                with (args.output/'results.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                (args.output/'progress.json').write_text(json.dumps(dict(completed_configurations=len(timings),total_configurations=len(args.segments)*len(args.waves))))
    report=dict(complete=True,condition_count=len(rows),kernel_count=len(timings)*1800,positions_per_kernel=len(centers),
        elapsed_including_inputs_verification_writing_s=time.perf_counter()-wall,device=str(device),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,compiled=args.compile_gpu,
        height_scaling_max_error_m=scale_error,checks=checks,timings=timings,
        timing_note=('Includes compilation on the first occurrence of a shape; pre-existing compiler caches may be reused. Not a clean-cache benchmark.'
                     if args.compile_gpu else 'Eager Torch correctness run, including inputs, verification and writing. Not a paired performance benchmark.'))
    (args.output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','timings']}),flush=True)


if __name__=='__main__':main()

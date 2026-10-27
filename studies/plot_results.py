"""Regenerate the four scientific figures from retained CSV/kernel arrays."""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=ROOT/'results/reference_50x36.csv')
    p.add_argument('--kernel-dir',type=Path,default=ROOT/'results/kernel_energy')
    p.add_argument('--output',type=Path,default=ROOT/'runs/figures')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with args.csv.open() as f:rows=list(csv.DictReader(f))
    data={(int(r['segments']),float(r['wave']),float(r['height_scale'])):float(r['R_gait_deg']) for r in rows}
    refs={float(r['height_scale']):float(r['R_reference_deg']) for r in rows}
    waves=[1.5,2.,2.5,3.,3.5,4.];counts=[11,13,15,17,19];scales=np.arange(1,10)/4
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(args.output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    cmap=plt.get_cmap('viridis');norm=Normalize(.25,2.25)
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True,constrained_layout=True)
    for s in scales:
        axs[0].plot(waves,[data[19,w,s] for w in waves],'-o',color=cmap(norm(s)),label=f'Height ×{s:g}')
        axs[1].plot(counts,[data[n,3.,s] for n in counts],'-o',color=cmap(norm(s)))
    axs[0].set(xlabel='Body wave number',ylabel='Gait-conditioned pair-angle RMS (deg)',title='19 leg-bearing segments',xticks=waves)
    axs[1].set(xlabel='Leg-bearing segments',title='Wave number 3',xticks=counts)
    for ax in axs:ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',fontsize=9,ncol=5,frameon=False)
    fig.suptitle('Geometric trends across terrain height amplitudes')
    save(fig,'01_wave_and_segment_trends')

    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True,constrained_layout=True)
    for n,ax in zip(counts,axs.flat):
        for w in waves:ax.plot([refs[s] for s in [0.,*scales]],[data[n,w,s] for s in [0.,*scales]],'-o',ms=3,label=f'Wave {w:g}')
        ax.set_title(f'{n} leg-bearing segments');ax.grid(alpha=.2)
    axs[1,2].axis('off');axs[1,2].legend(*axs[0,0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.supxlabel('Neutral-reference terrain roughness (deg)');fig.supylabel('Gait-conditioned pair-angle RMS (deg)')
    save(fig,'02_reference_vs_gait_conditioned')

    fig,axs=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    for s,ax in zip(scales,axs.flat):
        values=np.array([[data[n,w,s] for w in waves] for n in counts]);im=ax.imshow(values,origin='lower',aspect='auto',cmap='viridis')
        ax.set(xticks=range(6),xticklabels=[f'{w:g}' for w in waves],yticks=range(5),yticklabels=counts,title=f'Height ×{s:g}',xlabel='Wave number',ylabel='Segments')
        for y in range(5):
            for x in range(6):ax.text(x,y,f'{values[y,x]:.2f}',ha='center',va='center',fontsize=9,color='white' if im.norm(values[y,x])<.5 else 'black')
        fig.colorbar(im,ax=ax,label='RMS (deg)')
    fig.suptitle('Complete wave-number × segment-count results; independent panel color scales')
    save(fig,'03_full_cross_results')

    with np.load(args.kernel_dir/'N19_wave3.npz') as z:energy=z['energy']
    assert energy.shape==(1800,9)
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for si,ax in zip([0,8],axs):
        values=np.degrees(np.sqrt(energy[:,si])).reshape(50,36)
        im=ax.imshow(values,origin='lower',aspect='auto',extent=[-5,355,-.01,.99])
        ax.set(title=f'Height ×{scales[si]:g}',xlabel='Heading (deg)',ylabel='Cycle phase')
        fig.colorbar(im,ax=ax,label='Per-kernel spatial pair-angle RMS (deg)')
    fig.suptitle('Phase and heading kernels: wave 3, 19 segments')
    save(fig,'04_phase_heading_kernels')
    print(f'Wrote four figures in PNG and SVG to {args.output}')


if __name__=='__main__':main()

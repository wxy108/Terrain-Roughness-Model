import csv,hashlib,json
import numpy as np
import pytest
import torch
from terrain_roughness import calculate_roughness


def test_included_data_hashes(root):
    m=json.loads((root/'data/manifest.json').read_text())
    for record in m['records']+[m['full_base'],m['gait_bank']]:
        assert hashlib.sha256((root/record['path']).read_bytes()).hexdigest()==record['sha256']


def test_all_saved_energy_aggregates(root):
    with (root/'results/reference_50x36.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==300
    for n in [11,13,15,17,19]:
        for wave in [1.5,2.,2.5,3.,3.5,4.]:
            with np.load(root/f'results/kernel_energy/N{n}_wave{wave:g}.npz') as z:e=z['energy']
            assert e.shape==(1800,9) and np.isfinite(e).all()
            r=np.r_[0.,np.degrees(np.sqrt(e.mean(0)))]
            subset=[x for x in rows if int(x['segments'])==n and float(x['wave'])==wave]
            np.testing.assert_allclose(r,[float(x['R_gait_deg']) for x in subset],atol=1e-12,rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("segments,wave,scale",[(19,3.,1.),(11,1.5,.25),(15,1.5,2.25)])
def test_complete_real_terrain(root,segments,wave,scale):
    with (root/'results/reference_50x36.csv').open() as f:
        expected=next(float(r['R_gait_deg']) for r in csv.DictReader(f) if
            (int(r['segments']),float(r['wave']),float(r['height_scale']))==(segments,wave,scale))
    label=f'{scale:.2f}'.replace('.','p')
    with np.load(root/f'data/heightmaps/libo_x{label}.npz') as z:
        value=calculate_roughness(z['height'],wave,segments=segments,grid_spacing_m=float(z['dx_m']),map_origin_xy_m=z['origin_xy_m'],device='auto')
    assert value==pytest.approx(expected,abs=1e-5)


@pytest.mark.compiled
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable.')
def test_compiled_actual_terrain(root):
    with np.load(root/'data/heightmaps/libo_x1p00.npz') as z:
        value=calculate_roughness(z['height'],3.,grid_spacing_m=float(z['dx_m']),map_origin_xy_m=z['origin_xy_m'],compile_gpu=True,device='cuda')
    assert value==pytest.approx(9.43447228581486,abs=1e-5)

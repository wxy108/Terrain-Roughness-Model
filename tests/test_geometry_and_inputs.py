import numpy as np
import pytest
from terrain_roughness import calculate_roughness
from terrain_roughness.core import _kernels, _scan_centers


@pytest.mark.parametrize("segments",[11,13,15,17,19])
@pytest.mark.parametrize("wave",[1.5,2.,2.5,3.,3.5,4.])
def test_saved_kernel_bank(segments,wave):
    feet,d,i,j,bank=_kernels(wave,segments)
    assert feet.shape==(1800,segments,2)
    assert d.shape==(1800,segments*(segments-1)//2)
    assert (d>0).all()
    assert np.all(feet.min((0,1))>=bank['common_foot_min_xy_m']-1e-14)
    assert np.all(feet.max((0,1))<=bank['common_foot_max_xy_m']+1e-14)
    assert len(i)==len(j)==segments*(segments-1)//2


def test_study_grid_and_valid_boundaries():
    _,_,_,_,bank=_kernels(3.,19)
    c=_scan_centers((869,1089),np.array([.0005,.0005]),np.array([-.223,-.353]),None,.002,bank)
    assert len(c)==36300
    assert len(np.unique(c[:,0]))==220 and len(np.unique(c[:,1]))==165
    assert np.max(np.diff(np.unique(c[:,0])))<=.002+1e-14
    assert np.max(np.diff(np.unique(c[:,1])))<=.002+1e-14
    np.testing.assert_allclose(c.min(0)+bank['common_foot_min_xy_m'],[-.223,-.353],atol=1e-14)
    np.testing.assert_allclose(c.max(0)+bank['common_foot_max_xy_m'],[.321,.081],atol=1e-14)


@pytest.mark.parametrize("change",[
    dict(wave_number=2.2),dict(segments=19.1),dict(segments=10),dict(grid_spacing_m=-.001),
    dict(scan_step_m=0),dict(kernel_batch=0),dict(position_batch=-1),dict(device='mps'),
    dict(compile_gpu=True),dict(terrain_bounds_m=(-1,1,-1,1)),
])
def test_invalid_arguments_even_on_flat(change):
    args=dict(height_map=np.zeros((150,150)),wave_number=3.,grid_spacing_m=.001,device='cpu')
    args.update(change)
    with pytest.raises((ValueError,RuntimeError)):
        calculate_roughness(**args)


@pytest.mark.parametrize("height",[np.full((150,150),np.nan),np.zeros((2,)),np.zeros((1,150)),np.zeros((10,10))])
def test_invalid_heightmap(height):
    with pytest.raises(ValueError):calculate_roughness(height,3.,grid_spacing_m=.001,device='cpu')

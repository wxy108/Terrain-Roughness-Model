import numpy as np
import pytest
import torch
from terrain_roughness import calculate_roughness
from terrain_roughness.core import _PairEnergy, _kernels


def plane():
    dx, dy = .0008, .0011
    origin = np.array([-.09, -.08])
    x = origin[0] + np.arange(221)*dx
    y = origin[1] + np.arange(168)*dy
    xx, yy = np.meshgrid(x, y)
    return .12*xx-.08*yy+.001, dict(grid_spacing_m=(dx,dy), map_origin_xy_m=origin, scan_step_m=.04)


def analytic_plane(wave=3., segments=19):
    feet, d, i, j, _ = _kernels(wave, segments)
    delta = feet[:,i]-feet[:,j]
    angles = np.arctan(abs(.12*delta[:,:,0]-.08*delta[:,:,1])/d)
    return float(np.degrees(np.sqrt(np.mean(angles**2))))


@pytest.mark.parametrize("wave,segments", [(1.5,11),(3.,19),(4.,15)])
def test_planar_heightfield(wave, segments):
    height, kwargs = plane()
    value = calculate_roughness(height, wave, segments=segments, device="cpu", position_batch=3, **kwargs)
    assert value == pytest.approx(analytic_plane(wave, segments), abs=1e-5)


def test_chunk_sizes_and_vertical_offset():
    height, kwargs = plane()
    a = calculate_roughness(height,3.,device="cpu",kernel_batch=3,position_batch=7,**kwargs)
    b = calculate_roughness(height+.06,3.,device="cpu",kernel_batch=4,position_batch=2,**kwargs)
    assert a == pytest.approx(b,abs=1e-7)


def test_flat_ground():
    height, kwargs = plane()
    assert calculate_roughness(np.full_like(height,.02),3.,device="cpu",**kwargs) == 0.


def test_triangle_interpolation():
    # Triangles give heights .25 and .75, hence 45 deg. Bilinear gives a
    # different answer, so this checks the required interpolation convention.
    module = _PairEnergy(np.array([[0.,0.],[0.,1.]]),[1.,1.],[0.,0.],np.array([0]),np.array([1]),"cpu")
    with torch.inference_mode():
        f=torch.tensor([[[.25,.75],[.75,.75]]],dtype=torch.float64)
        e=module(f,torch.tensor([[.5]],dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64))
        value=float(torch.rad2deg(torch.sqrt(e))[0])
    assert value == pytest.approx(45.,abs=1e-5)


def test_last_grid_vertex():
    module = _PairEnergy(np.array([[0.,0.],[0.,1.]]),[1.,1.],[0.,0.],np.array([0]),np.array([1]),"cpu")
    with torch.inference_mode():
        f=torch.tensor([[[1.,1.],[0.,0.]]],dtype=torch.float64)
        e=module(f,torch.tensor([[np.sqrt(2)]],dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64))
    assert float(torch.rad2deg(torch.sqrt(e))[0]) == pytest.approx(np.degrees(np.arctan(1/np.sqrt(2))),abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA is unavailable.")
def test_cuda_plane():
    height, kwargs = plane()
    value=calculate_roughness(height,3.,device="cuda",kernel_batch=3,position_batch=7,**kwargs)
    assert value == pytest.approx(analytic_plane(),abs=1e-5)

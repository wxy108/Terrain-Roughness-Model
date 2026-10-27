"""Gait-conditioned terrain roughness for the saved centipede geometry.

The package includes gait_bank.npz. Requires numpy and torch.

    from terrain_roughness import calculate_roughness
    R_deg = calculate_roughness(height_m, 3.0, grid_spacing_m=0.0005)

All physical inputs use METERS. height_m[row, col] increases along +y, +x.
The result is degrees, not height RMS or neutral-configuration roughness.
Supported saved waves: 1.5, 2, 2.5, 3, 3.5, 4; segments: 11, 13, 15, 17, 19.
Their original calibrated amplitudes are retained; no gait interpolation.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn


@lru_cache(maxsize=1)
def _gait_bank():
    path = Path(__file__).with_name("gait_bank.npz")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _kernels(wave_number, segments):
    bank = _gait_bank()
    matches = np.flatnonzero(np.isclose(bank["waves"], wave_number, rtol=0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError(f"wave_number must be one of {bank['waves'].tolist()}.")
    if segments not in bank["segments"]:
        raise ValueError(f"segments must be one of {bank['segments'].tolist()}.")
    feet = bank[f"feet_{segments}_m"][matches[0]]
    mask = bank[f"stance_{segments}"][matches[0]]
    stance = np.stack([xy[contact] for xy, contact in zip(feet, mask)])
    angle = np.deg2rad(bank["headings_deg"])
    rotation = np.stack([np.stack([np.cos(angle), -np.sin(angle)], -1),
                         np.stack([np.sin(angle), np.cos(angle)], -1)], -2)
    # [phase, heading, stance foot, xy] -> 1,800 contact kernels.
    xy = np.einsum("pni,hji->phnj", stance, rotation).reshape(-1, segments, 2)
    i, j = np.triu_indices(segments, 1)
    distance = np.linalg.norm(xy[:, i] - xy[:, j], axis=-1)
    if np.any(distance <= 0):
        raise ValueError("A stance-foot pair has zero horizontal distance.")
    return xy, distance, i, j, bank


class _PairEnergy(nn.Module):
    """Mean squared pair angles for a batch of kernels and translation points."""

    def __init__(self, height, spacing, origin, pair_i, pair_j, device):
        super().__init__()
        self.register_buffer("height", torch.as_tensor(height, dtype=torch.float64, device=device))
        self.register_buffer("spacing", torch.as_tensor(spacing, dtype=torch.float64, device=device))
        self.register_buffer("origin", torch.as_tensor(origin, dtype=torch.float64, device=device))
        self.register_buffer("pair_i", torch.as_tensor(pair_i, device=device))
        self.register_buffer("pair_j", torch.as_tensor(pair_j, device=device))

    def _slopes(self, feet, distance, centers):
        xy = (centers[None, :, None, :] + feet[:, None, :, :] - self.origin) / self.spacing
        # Clamp the cell index, not the queried position: the last grid vertex
        # belongs to the last cell with fractional coordinate 1.
        i = xy[..., 0].floor().long().clamp(0, self.height.shape[1] - 2)
        j = xy[..., 1].floor().long().clamp(0, self.height.shape[0] - 2)
        u, v = xy[..., 0] - i, xy[..., 1] - j
        h = self.height
        # Same diagonal triangulation as the validated simulation heightfield.
        z = (h[j, i] * (1 - torch.maximum(u, v))
             + h[j + 1, i + 1] * torch.minimum(u, v)
             + torch.where(u >= v, h[j, i + 1], h[j + 1, i]) * abs(u - v))
        return (abs(z[:, :, self.pair_i] - z[:, :, self.pair_j]) / distance[:, None, :]).float()

    def forward(self, feet, distance, centers):
        beta = torch.atan(self._slopes(feet, distance, centers))
        return beta.square().sum((1, 2), dtype=torch.float64) / self.pair_i.numel()


def _scan_centers(shape, spacing, origin, terrain_bounds_m, scan_step_m, bank):
    extent = origin + spacing * (np.array(shape[::-1]) - 1)
    bounds = np.array([origin[0], extent[0], origin[1], extent[1]])
    if terrain_bounds_m is not None:
        bounds = np.asarray(terrain_bounds_m, dtype=np.float64)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or np.any(bounds[[0, 2]] < origin) or np.any(bounds[[1, 3]] > extent)
                or np.any(bounds[[0, 2]] > bounds[[1, 3]])):
            raise ValueError("terrain_bounds_m must be ordered (xmin, xmax, ymin, ymax) within the map.")
    lower = bounds[[0, 2]] - bank["common_foot_min_xy_m"]
    upper = bounds[[1, 3]] - bank["common_foot_max_xy_m"]
    if np.any(upper < lower):
        raise ValueError("Terrain is too small for the common valid footprint domain (about 106.49 mm per axis).")
    grid_size = np.ceil((upper - lower) / scan_step_m).astype(int) + 1
    x = np.linspace(lower[0], upper[0], grid_size[0])
    y = np.linspace(lower[1], upper[1], grid_size[1])
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def calculate_roughness(
    height_map,
    wave_number,
    *,
    grid_spacing_m,
    segments=19,
    scan_step_m=0.002,
    map_origin_xy_m=(0.0, 0.0),
    terrain_bounds_m=None,
    device="auto",
    compile_gpu=False,
    kernel_batch=None,
    position_batch=None,
):
    """Return the gait-conditioned, complete-graph pair-angle RMS in DEGREES.

    Parameters
    ----------
    height_map : 2-D array, shape [ny, nx]
        Actual terrain heights in meters; finite values, at least 2 x 2.
        A grayscale picture requires calibration to physical heights first.
    wave_number : float
        Select one saved gait: 1.5, 2, 2.5, 3, 3.5 or 4.
    grid_spacing_m : float or (dx, dy)
        Physical spacing between height samples, in meters. Required because
        horizontal distances and height differences define the pair slopes.
    segments : int
        Leg-bearing segment count, default 19. Selects the saved geometry.
    scan_step_m : float
        Maximum translation-grid spacing, default 0.002 m. Both endpoints
        are included; actual grid spacings can be slightly smaller.
    map_origin_xy_m : (x0, y0)
        Physical xy coordinate of height_map[0, 0], default (0, 0).
    terrain_bounds_m : (xmin, xmax, ymin, ymax) or None
        Optional scan terrain rectangle within the map. None uses the full
        map extent. Use the old simulation ROI to reproduce previous values.
    device : 'auto', 'cpu', 'cuda', or 'cuda:N'
        Auto uses CUDA when available. CPU uses the same Torch operations.
    compile_gpu : bool
        Optional torch.compile acceleration on CUDA. Requires a working
        Torch/Triton compiler; first compilation adds setup time. Default
        False runs without compiler setup. Compilation errors are explicit.
    kernel_batch, position_batch : positive int or None
        Memory controls only. Chunking retains every kernel and position.

    Definition
    ----------
    R = (180/pi) * sqrt(mean_phase,heading,position,pair[
                            atan(abs(delta_height) / horizontal_distance)^2]))

    The bank has 50 phases [0, .02, ..., .98] and 36 headings [0, 10, ..., 350]
    degrees. Only stance feet enter the pair statistic. ALL projected feet,
    including swing feet, define one common valid rectangle across all 30
    saved wave/segment combinations. This preserves the comparison protocol.
    A single final square root follows averaging; local RMS values are not
    averaged. No boundary padding, foot rasterization or small-angle shortcut.
    """
    height = np.asarray(height_map, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 2 or not np.isfinite(height).all():
        raise ValueError("height_map must be a finite 2-D array of at least 2 x 2 heights in meters.")
    spacing = np.asarray(grid_spacing_m, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 2)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("grid_spacing_m must be positive: a scalar or (dx, dy), in meters.")
    origin = np.asarray(map_origin_xy_m, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("map_origin_xy_m must be finite (x0, y0), in meters.")
    if not np.isfinite(scan_step_m) or scan_step_m <= 0:
        raise ValueError("scan_step_m must be positive.")
    if not isinstance(segments, (int, np.integer)) or isinstance(segments, bool):
        raise ValueError("segments must be an integer.")
    feet, distance, pair_i, pair_j, bank = _kernels(wave_number, segments)

    centers = _scan_centers(height.shape, spacing, origin, terrain_bounds_m, scan_step_m, bank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This implementation supports CPU and CUDA devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'.")
    if compile_gpu and device.type != "cuda":
        raise ValueError("compile_gpu=True requires a CUDA device.")
    kb = kernel_batch if kernel_batch is not None else (4 if device.type == "cuda" else 1)
    pb = position_batch if position_batch is not None else (16384 if device.type == "cuda" else 1024)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0 for v in (kb, pb)):
        raise ValueError("kernel_batch and position_batch must be positive integers.")
    if np.ptp(height) == 0:
        return 0.0  # Exact constant-height control after validating every argument.
    with torch.inference_mode():
        module = _PairEnergy(height, spacing, origin, pair_i, pair_j, device)
        evaluate = torch.compile(module, fullgraph=True, dynamic=False) if compile_gpu else module
        feet_t = torch.as_tensor(feet, dtype=torch.float64, device=device)
        distance_t = torch.as_tensor(distance, dtype=torch.float64, device=device)
        centers_t = torch.as_tensor(centers, dtype=torch.float64, device=device)
        energy = torch.zeros(len(feet), dtype=torch.float64, device=device)
        for a in range(0, len(feet), kb):
            b = min(a + kb, len(feet))
            for p in range(0, len(centers), pb):
                # Each partial sum retains its actual point count. The final
                # short chunk has the same per-position weight as every other.
                energy[a:b] += evaluate(feet_t[a:b], distance_t[a:b], centers_t[p:p+pb])
        mean_square = energy.sum() / (len(feet) * len(centers))
        return float(torch.rad2deg(torch.sqrt(mean_square)).cpu())

# core.py: a line-by-line mathematical walkthrough

This is a manual-review companion to the main algorithm in [`src/terrain_roughness/core.py`](../src/terrain_roughness/core.py). Read each source block, its equations and its line table together. The complete nonblank source is reproduced below in its original order.

**Reviewed revision:** [`7d96fc3`](https://github.com/wxy108/Terrain-Roughness-Model/tree/7d96fc3f2a4a838079e73f1660e927c2b780c8e5). **Source blob:** `e21aab1af11c41d0b951585e71782f89853c97f9`.

Every source link and line number in this document is pinned to that revision. Code blocks are copied verbatim from that blob; they are not pseudocode. Tables explain executable statements individually, grouping multiline statements where needed. Comments, blank separators and the long API docstring are identified separately. This walkthrough documents the existing implementation; it does not change or newly validate the full terrain study.

For the repository-wide file map, see the [code guide](CODE_GUIDE.md). For the original derivation, see the [method document](METHOD.md).

## 1. The quantity being calculated

Let the calibrated height map represent a piecewise-linear surface `h(x,y)` in meters. A saved gait supplies horizontal foot positions at `P=50` phases. Rotating each phase through `T=36` headings gives `K=P*T=1800` kernels.

For this bank, a selected configuration with `N` leg-bearing segments has exactly `N` stance feet per phase. Its number of unordered stance-foot pairs is `M=N*(N-1)/2`. At `N=19`, `M=171`. Let `L` be the number of valid translation centers; the included scan-domain terrain has `L=36300`.

For kernel `k`, center `ell` and stance pair `(i,j)`, define:

Here `k` indexes a phase/heading kernel, not the body wave-number parameter. The selected wave number and segment count are fixed within one call.

```math
\begin{aligned}
\mathbf q_{k\ell i}&=\mathbf c_\ell+\mathbf f_{ki},\\
z_{k\ell i}&=h(\mathbf q_{k\ell i}),\\
d_{kij}&=\|\mathbf f_{ki}-\mathbf f_{kj}\|_2,\\
s_{k\ell ij}&=\frac{|z_{k\ell i}-z_{k\ell j}|}{d_{kij}},\\
\beta_{k\ell ij}&=\arctan(s_{k\ell ij}).
\end{aligned}
```

The intended discrete statistic is:

```math
R_g=\frac{180}{\pi}\sqrt{
\frac{1}{KLM}\sum_{k=1}^{K}\sum_{\ell=1}^{L}
\sum_{i\lt j}\beta_{k\ell ij}^{2}}.
```

Positions, heights and distances use **meters**. Slopes are dimensionless, `atan` returns radians, and the final output is **degrees**. The source's mixed-precision evaluation is described at lines 73, 76 and 77 below. This statistic averages over a finite grid, phases, headings and foot pairs; it is not a continuous-area integral.

### Tensor dimensions used below

`B_k` and `B_l` denote the current kernel-batch and translation-batch sizes.

| Variable | Shape | Meaning |
|---|---|---|
| `height` | `[ny, nx]` | Physical terrain heights. |
| `spacing`, `origin` | `[2]` | `(dx, dy)` and `(x0, y0)` in meters. |
| `stance` | `[P, N, 2]` | Selected stance-foot xy coordinates before heading rotation. |
| `rotation` | `[T, 2, 2]` | Heading matrices. |
| `feet` / full `xy` from `_kernels` | `[K, N, 2]` | Rotated stance-foot positions relative to the translation center. |
| `pair_i`, `pair_j` | `[M]` | The two endpoint indices of each unordered pair. |
| `distance` | `[K, M]` | Pairwise horizontal distances. |
| `centers` | `[L, 2]` | Translation-center coordinates. |
| `xy` inside `_slopes` | `[B_k, B_l, N, 2]` | Continuous height-map coordinates of translated feet. |
| `z` | `[B_k, B_l, N]` | Interpolated terrain height under each foot. |
| slope / `beta` | `[B_k, B_l, M]` | Pair slopes / pair angles for the current batch. |
| `forward()` result | `[B_k]` | Partial energy summed over the current positions and averaged over pairs. |
| `energy` | `[K]` | Accumulated energy across all positions, one value per kernel. |

## 2. File contract and imports — lines 1–18

[Source at line 1](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L1) · original lines 1–18

```python
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
```

| Source lines | Meaning for review |
|---|---|
| 1–12 | Module docstring: defines the physical units, output quantity, supported saved configurations and preserved amplitudes. These lines do not execute a terrain calculation. |
| 13 | Imports a cache decorator used to avoid repeatedly reading the same gait-bank file. |
| 14 | Imports filesystem-path handling. |
| 16 | Imports NumPy for input validation, geometry, pair indices and translation-grid construction. |
| 17 | Imports Torch for array evaluation on CPU or CUDA. |
| 18 | Imports `nn.Module`, used as the container that can optionally be compiled. |

## 3. Load the saved geometry — lines 21–25

[Source at line 21](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L21) · original lines 21–25

```python
@lru_cache(maxsize=1)
def _gait_bank():
    path = Path(__file__).with_name("gait_bank.npz")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}
```

| Source line | Code meaning and mathematical role |
|---|---|
| 21 | Cache the result of this no-argument function. It changes file-loading cost, not the statistic. |
| 22 | Defines the bank loader. |
| 23 | Locates `gait_bank.npz` beside `core.py`; no external simulation installation is used here. |
| 24 | Opens the lossless array archive with pickle loading disabled. |
| 25 | Materializes every array in a dictionary before the file context closes. |

**Review boundary:** this file consumes saved foot coordinates, stance masks and common bounds. It does not derive forward kinematics, choose calibrated amplitudes or infer stance from dynamics. Those are upstream properties of the packaged bank; its provenance and integrity checks are documented elsewhere in the repository.

## 4. Construct kernels and foot pairs — lines 28–47

[Source at line 28](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L28) · original lines 28–47

```python
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
```

For phase `p` and heading `q`, the rotation is:

```math
Q_q=\begin{bmatrix}\cos\theta_q&-\sin\theta_q\\
\sin\theta_q&\cos\theta_q\end{bmatrix},\qquad
f_{pqnj}=\sum_{a=1}^{2}Q_{qja}r_{pna}.
```

This is the coordinate contraction implemented by `einsum("pni,hji->phnj", ...)`: the repeated `i` sums over input xy coordinates, while `j` is the output coordinate. After reshaping, the flattened kernel index is `k=p*T+q` in zero-based indexing. For valid saved inputs, rotation preserves pair distances apart from floating-point roundoff.

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 28 | Accept the selected wave and segment count. |
| 29 | Load the saved bank. |
| 30 | Find saved wave entries within absolute tolerance `1e-12`, with zero relative tolerance. This selects a stored gait rather than interpolating gaits. |
| 31–32 | Require exactly one matching wave; otherwise raise an error. |
| 33–34 | Require a saved segment count. The public function separately checks that its type is an integer. |
| 35 | Select all stored phase-dependent foot xy coordinates for this configuration. |
| 36 | Select the corresponding Boolean stance masks. |
| 37 | At each phase, retain only feet whose mask is true. Stack them into `[P,N,2]`; equal stance counts are a property of the saved bank. |
| 38 | Convert stored heading angles from degrees to radians for sine and cosine. |
| 39–40 | Construct the `[T,2,2]` matrices `Q_q`. |
| 41 | Comment identifying the dimensions before flattening. |
| 42 | Apply every heading matrix to every phase's stance feet, then flatten phase and heading into `K=1800` kernels. |
| 43 | Generate all upper-triangle pairs with offset 1: each `i<j` appears once; self-pairs and duplicate reversed pairs are excluded. |
| 44 | Compute `d_kij = sqrt((fx_i-fx_j)^2 + (fy_i-fy_j)^2)` in meters for each kernel and pair. |
| 45–46 | Reject a zero or negative distance before division. Small positive distances are retained; there is no fitted pair-distance cutoff. |
| 47 | Return the rotated feet, distances, pair indices and bank. |

**Hand check:** rotating `(0.01,0)` m through 90° gives `(0,0.01)` m. Translating by center `(0.02,0.03)` m then gives sample position `(0.02,0.04)` m. This checks rotation direction and the sign of the translation.

## 5. Store fixed arrays on the execution device — lines 50–59

[Source at line 50](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L50) · original lines 50–59

```python
class _PairEnergy(nn.Module):
    """Mean squared pair angles for a batch of kernels and translation points."""

    def __init__(self, height, spacing, origin, pair_i, pair_j, device):
        super().__init__()
        self.register_buffer("height", torch.as_tensor(height, dtype=torch.float64, device=device))
        self.register_buffer("spacing", torch.as_tensor(spacing, dtype=torch.float64, device=device))
        self.register_buffer("origin", torch.as_tensor(origin, dtype=torch.float64, device=device))
        self.register_buffer("pair_i", torch.as_tensor(pair_i, device=device))
        self.register_buffer("pair_j", torch.as_tensor(pair_j, device=device))
```

| Source line | Code meaning and mathematical role |
|---|---|
| 50 | Define the Torch module used to evaluate partial pair energies. |
| 51 | Descriptive docstring. Despite the shorthand name, the exact normalization of the return value is specified at line 77. |
| 53 | Constructor inputs: terrain, coordinate metadata, pair endpoints and CPU/CUDA device. |
| 54 | Initialize the parent Torch module. |
| 55 | Register the height map as a float64 buffer on the selected device. |
| 56 | Register the xy grid spacing as a float64 buffer. |
| 57 | Register the physical map origin as a float64 buffer. |
| 58 | Register the first endpoint index of each pair. The source arrays contain integer indices. |
| 59 | Register the second endpoint indices. |

Buffers are fixed model data, not trainable parameters. These lines establish storage and precision; they do not fit a physical model.

## 6. Translate feet, interpolate heights and form slopes — lines 61–73

[Source at line 61](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L61) · original lines 61–73

```python
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
```

### 6.1 Physical position to continuous grid coordinates

Line 62 implements componentwise conversion:

```math
\xi=\frac{c_x+f_x-x_0}{d_x},\qquad
\eta=\frac{c_y+f_y-y_0}{d_y}.
```

`centers[None,:,None,:]` has shape `[1,B_l,1,2]` and `feet[:,None,:,:]` has shape `[B_k,1,N,2]`. Broadcasting adds each center to every foot in every current kernel. These coordinates stay continuous; feet are not rounded to pixels.

The cell column and row are `i=floor(xi)` and `j=floor(eta)`, with local coordinates `u=xi-i`, `v=eta-j`. Here the local names `i,j` mean **grid indices**. They are distinct from the foot-pair indices stored as `self.pair_i` and `self.pair_j`.

### 6.2 Triangle interpolation

Write the four cell heights as `h00=h[j,i]`, `h10=h[j,i+1]`, `h01=h[j+1,i]` and `h11=h[j+1,i+1]`. The diagonal runs from `(0,0)` to `(1,1)`:

```math
z(u,v)=h_{00}[1-\max(u,v)]+h_{11}\min(u,v)+
\begin{cases}
h_{10}(u-v),&u\geq v,\\
h_{01}(v-u),&u\lt v.
\end{cases}
```

For `u>=v`, the weights are `(1-u, u-v, v)` on `(h00,h10,h11)`. For `u<v`, they are `(1-v, v-u, u)` on `(h00,h01,h11)`. In a valid cell they are nonnegative and sum to one.

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 61 | Accept one batch of rotated feet, pair distances and translation centers. |
| 62 | Form physical sample positions, subtract the map origin and divide by xy spacing; output `[B_k,B_l,N,2]`. |
| 63–64 | Boundary-convention comments: clamp the cell index, not the sample position. |
| 65 | Choose the cell column, restricting it to `0..nx-2` so that `i+1` remains a valid vertex. |
| 66 | Choose the cell row, restricting it to `0..ny-2` so that `j+1` remains valid. |
| 67 | Compute continuous fractional coordinates after selecting the cell. A sample at the final x vertex has `u=1`, not a clamped position at an earlier vertex. |
| 68 | Alias the height tensor as `h`. |
| 69 | Comment identifying the required triangulation convention. |
| 70 | Add the `h00` contribution with weight `1-max(u,v)`. |
| 71 | Add the `h11` contribution with weight `min(u,v)`. |
| 72 | Select the third triangle vertex (`h10` or `h01`) and apply its weight `abs(u-v)`. The completed `z` tensor is `[B_k,B_l,N]`. |
| 73 | Select each pair's two sampled heights, take the absolute difference and divide by `distance[:,None,:]`. Output `[B_k,B_l,M]`, then cast the slopes to float32. |

**Boundary dependence:** clipping cell indices is not a replacement for a valid scan domain. The intended samples stay inside the terrain because `_scan_centers()` uses the bank's common bounds. Line 65 does not independently clamp an out-of-domain physical sample to the terrain edge.

**Precision:** positions, interpolation and distances use float64. Line 73 explicitly casts the resulting slopes to float32, matching the retained implementation. The angle and square below are therefore evaluated in float32 before float64 accumulation.

## 7. Compute a partial energy — lines 75–77

[Source at line 75](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L75) · original lines 75–77

```python
    def forward(self, feet, distance, centers):
        beta = torch.atan(self._slopes(feet, distance, centers))
        return beta.square().sum((1, 2), dtype=torch.float64) / self.pair_i.numel()
```

For one kernel and the current set of translation indices `C`, the returned value is:

```math
F_{k,C}=\frac{1}{M}\sum_{\ell\in C}\sum_{i\lt j}
\left[\arctan(s_{k\ell ij})\right]^2.
```

| Source line | Code meaning and mathematical role |
|---|---|
| 75 | Define one batch evaluation. |
| 76 | Apply the full `atan` to every nonnegative pair slope; the result is in radians. No small-slope approximation is used. |
| 77 | Square in the angle tensor's precision, accumulate over position dimension 1 and pair dimension 2 in float64, and divide by `M=self.pair_i.numel()`. Keep kernel dimension 0. |

The output is **a sum over the current positions**, not an average over positions and not an RMS. This distinction is what makes unequal final position chunks receive the correct weight later. A float64 sum does not undo the float32 rounding of each angle and square.

## 8. Construct the shared valid scan grid — lines 80–97

[Source at line 80](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L80) · original lines 80–97

```python
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
```

For one axis, all terrain samples must obey `a <= center + foot <= b`. If the bank's precomputed foot extremes are `f_min` and `f_max`, the common center interval is:

```math
a-f_{\min}\ \leq\ c\ \leq\ b-f_{\max}.
```

This explains the two minus signs at lines 89–90. The bank bounds cover all saved wave/segment combinations, phases and headings, including swing-foot projections. Only stance feet contribute to pair energies; all-foot bounds define a common comparison area. This function consumes those precomputed extremes rather than recomputing them from the current gait.

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 80 | Accept map dimensions, physical metadata, optional terrain ROI, maximum scan spacing and the bank. |
| 81 | Reverse `[ny,nx]` to `[nx,ny]`, subtract one interval from each point count, multiply by spacing, then add the origin to obtain the final map vertex `(xmax,ymax)`. |
| 82 | Default terrain bounds are `(x0,xmax,y0,ymax)`. |
| 83 | Enter the optional ROI branch if a terrain rectangle was supplied. |
| 84 | Convert its four coordinates to float64. |
| 85–88 | Require shape `[4]`, finite values, lower/upper corners inside the map and nondecreasing bounds in each axis; otherwise raise an error. |
| 89 | Compute each center-axis lower bound: terrain lower corner minus minimum foot offset. |
| 90 | Compute each center-axis upper bound: terrain upper corner minus maximum foot offset. |
| 91–92 | Reject a terrain too small for this common footprint. The approximate 106.49 mm in the message describes this saved bank. |
| 93 | For each center interval of length `D`, choose `ceil(D/scan_step_m)+1` grid points. |
| 94 | Generate endpoint-inclusive x centers with `linspace`. |
| 95 | Generate endpoint-inclusive y centers. |
| 96 | Form their Cartesian product with `meshgrid`. |
| 97 | Flatten the Cartesian product to `[L,2]`. With default meshgrid ordering, x varies fastest within each y row. |

For `D>0`, actual spacing is `D/ceil(D/s) <= s`. If a valid center interval has zero length, it contributes one point. The included terrain gives `220*165=36300` centers. Another map or ROI can give a different `L`; that is why `L` is obtained from `len(centers)` rather than hardcoded in the public function.

## 9. Public signature and API contract — lines 100–156

[Source at line 100](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L100) · original lines 100–156

```python
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
```

| Source line(s) | Argument or documentation | Meaning |
|---|---|---|
| 100 | `calculate_roughness` | Public orchestration function. |
| 101 | `height_map` | Calibrated heights `h[row,col]` in meters. |
| 102 | `wave_number` | Selects a stored gait and its saved calibrated amplitude. |
| 103 | `*` | All following arguments must be passed by keyword; it has no mathematical effect. |
| 104 | `grid_spacing_m` | Required physical `(dx,dy)` or common scalar spacing. |
| 105 | `segments=19` | Selects the saved leg-bearing segment configuration. In this bank it also equals the stance count used by `_kernels`. |
| 106 | `scan_step_m=0.002` | Maximum requested translation-grid spacing, 2 mm. This is different from the height-map pixel spacing. |
| 107 | `map_origin_xy_m` | Physical coordinate of the first height-map element. |
| 108 | `terrain_bounds_m` | Optional `(xmin,xmax,ymin,ymax)` ROI within the parent map. |
| 109 | `device` | CPU/CUDA execution selection; `auto` prefers available CUDA. |
| 110 | `compile_gpu` | Optional compilation of the pair-energy module. |
| 111–112 | `kernel_batch`, `position_batch` | Maximum chunk sizes; every kernel and center is still included. |
| 113 | End of signature | Begins the function body. |
| 114–156 | API docstring | Documents units, arguments, the RMS definition, the common-area convention and the final-root rule. These lines are documentation, not extra calculations. |

The detailed meanings of the long docstring are expanded by the equations and line tables in this walkthrough. Its references to bounds including swing feet and to one final square root describe essential protocol choices.

## 10. Validate and prepare mathematical inputs — lines 157–174

[Source at line 157](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L157) · original lines 157–174

```python
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
```

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 157 | Convert the height input to a NumPy float64 array. |
| 158–159 | Require a finite 2-D map with at least two samples per axis. |
| 160 | Convert horizontal spacing to float64. |
| 161–162 | Expand one scalar spacing to `[dx,dy]` with equal entries. |
| 163–164 | Require two finite positive spacings, so conversion to grid coordinates has valid physical denominators. |
| 165 | Convert the map origin to float64. |
| 166–167 | Require two finite origin coordinates. |
| 168–169 | Require a finite positive requested scan spacing. |
| 170–171 | Require an integer segment count and explicitly reject Boolean input. |
| 172 | Build the selected stance kernels, all unordered pair distances and pair indices. |
| 174 | Build the shared valid translation grid for this map and ROI. |

This stage also validates saved-gait availability and geometric coverage. A constant-height map does not bypass these input and geometry checks.

## 11. Execution settings and the exact flat case — lines 176–188

[Source at line 176](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L176) · original lines 176–188

```python
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
```

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 176 | Resolve `auto` to available CUDA or CPU; otherwise parse the requested Torch device. |
| 177–178 | Accept only CPU and CUDA device families. |
| 179–180 | Raise if CUDA was explicitly requested but is unavailable. |
| 181–182 | Require a CUDA device when GPU compilation is requested. This is an argument check, not a demonstration that a compiler can successfully run. |
| 183 | Default kernel-batch size: 4 on CUDA, 1 on CPU; otherwise use the provided size. |
| 184 | Default position-batch size: 16,384 on CUDA, 1,024 on CPU. |
| 185–186 | Require positive integer batch sizes and reject Boolean sizes. |
| 187 | Test whether the entire input height array has exactly zero peak-to-peak range. |
| 188 | Return exactly `0.0` after the preceding checks: every sampled height difference, slope and pair angle is zero. |

`np.ptp(height)==0` is an exact fast-path test, not a near-flat tolerance. A map whose ROI is flat but whose full input array is not constant may use the normal evaluation route. The intended metric does not depend on CPU/CUDA choice or chunk sizes; mixed precision and compiled ordering can produce small numerical differences, as described in the validation record.

## 12. Accumulate batches and return RMS — lines 189–203

[Source at line 189](https://github.com/wxy108/Terrain-Roughness-Model/blob/7d96fc3f2a4a838079e73f1660e927c2b780c8e5/src/terrain_roughness/core.py#L189) · original lines 189–203

```python
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
```

| Source line(s) | Code meaning and mathematical role |
|---|---|
| 189 | Disable gradient tracking for this evaluation. |
| 190 | Construct the pair-energy module with the fixed map and pair endpoints. |
| 191 | Use the module directly or wrap it with optional `torch.compile`. `fullgraph=True` and `dynamic=False` configure compilation, not a different RMS definition. |
| 192 | Move all rotated stance-foot coordinates to float64 tensors on the device. |
| 193 | Move pair distances in the same precision. |
| 194 | Move translation centers in the same precision. |
| 195 | Initialize one float64 accumulator per kernel; `len(feet)=K`. |
| 196 | Iterate over kernel-batch start indices. |
| 197 | Cap the current kernel batch at `K`, including a final shorter kernel batch. |
| 198 | Iterate over translation-batch start indices. |
| 199–200 | Comments explaining the unequal-final-chunk weighting. |
| 201 | Evaluate the current kernel/position slice and add its pair-averaged position sum to the corresponding kernel accumulators. Python slicing naturally includes only the remaining points in a short final batch. |
| 202 | Sum all kernels and divide by `K*L`. Pair normalization `1/M` was already applied inside `forward()`. |
| 203 | Take one square root, convert radians to degrees, transfer the scalar to CPU and return a Python `float`. |

After line 201 has visited every position chunk, let $E_k$ denote `energy[k]`.

```math
E_k
=\frac{1}{M}\sum_{\ell=1}^{L}\sum_{i\lt j}\beta_{k\ell ij}^{2}.
```

Line 202 computes `mean_square`, denoted by $S$.

```math
S
=\frac{1}{KL}\sum_{k=1}^{K}E_k
=\frac{1}{KLM}\sum_{k=1}^{K}\sum_{\ell=1}^{L}\sum_{i\lt j}\beta_{k\ell ij}^{2}.
```

There is no extra division by `M` at line 202. There is also no per-chunk square root before the sum. The equations describe normalization; the actual numerical values retain the precision choices identified earlier.

## 13. Small examples for hand verification

### Triangle height lookup and one pair angle

Use the two-by-two toy height map below with 1 m horizontal spacing and heights in meters; its size is chosen only for simple arithmetic:

```text
height = [[0, 0],
          [0, 1]]
origin = (0, 0)
foot 1 = (0.25, 0.75)
foot 2 = (0.75, 0.75)
one translation center = (0, 0)
```

The interpolation rule gives `z1=0.25 m` and `z2=0.75 m`. Horizontal distance is `0.5 m`, so `s=0.5/0.5=1`, `beta=atan(1)=pi/4`, and ideal one-pair RMS is **45 degrees**. Float32 angle/square evaluation can give a value slightly different from exactly 45.

Bilinear interpolation on this same grid would instead give `z1=0.1875`, `z2=0.5625`, slope `0.75` and angle about `36.8699 degrees`. This example makes the required triangle convention directly distinguishable. It exercises `_PairEnergy` in isolation; it is not a valid full-bank terrain for `calculate_roughness()`, whose common footprint requires a larger map.

### Unequal position chunks

Suppose one kernel and one pair produce angles `10, 20, 30 degrees` at three positions, split into chunks of two and one. Expressing squared angles in degrees only makes this hand calculation simpler; the source accumulates radians squared and converts at the end.

```text
first partial sum  = 10^2 + 20^2 = 500
second partial sum = 30^2 = 900
global mean square = (500 + 900) / 3
RMS                = sqrt(1400 / 3) = 21.602469 degrees
```

Giving the two chunks equal weight would be wrong because they contain different numbers of positions. Averaging their local RMS values would also be wrong. Lines 77 and 201–203 implement the global-sum construction above.

### A plane and a constant vertical offset

For `h(x,y)=a*x+b*y+c`, triangle interpolation reproduces the plane. A foot pair has:

```math
\Delta h=a\,\Delta f_x+b\,\Delta f_y.
```

Both the translation center and the constant `c` cancel. This gives an independent analytic reference for each rotated pair and explains why adding a constant height offset cannot change the ideal metric. Heading can still change the response because it rotates the foot-pair direction relative to the plane slope.

## 14. Existing tests corresponding to these modules

| Review item | Existing test and what it checks |
|---|---|
| Stance kernel shapes and pair counts | [`test_saved_kernel_bank`](../tests/test_geometry_and_inputs.py): all 30 saved wave/segment combinations, positive pair distances and containment within the common bounds. |
| Common grid and boundaries | [`test_study_grid_and_valid_boundaries`](../tests/test_geometry_and_inputs.py): 220×165 centers, endpoint relationships and maximum 2 mm spacing. |
| Physical inputs and flat-case validation | [`test_invalid_arguments_even_on_flat` and `test_invalid_heightmap`](../tests/test_geometry_and_inputs.py): invalid argument and height-map cases. |
| Analytic plane | [`test_planar_heightfield` and `test_cuda_plane`](../tests/test_metric.py): compare the calculated angle RMS against a plane-derived value on CPU and, when available, CUDA. |
| Triangle versus bilinear lookup | [`test_triangle_interpolation`](../tests/test_metric.py): the 45-degree example above. |
| Final grid vertex | [`test_last_grid_vertex`](../tests/test_metric.py): evaluate the upper boundary through the final cell. |
| Chunking and vertical offset | [`test_chunk_sizes_and_vertical_offset`](../tests/test_metric.py): compare calculations with different chunk sizes and a constant height shift. This test varies both together; it does not separately isolate their contributions. |
| Constant-height control | [`test_flat_ground`](../tests/test_metric.py): exact zero. |
| Saved data and numerical regressions | [`test_data_and_regression.py`](../tests/test_data_and_regression.py): file hashes, all retained energy aggregates, optional full real-terrain scans and optional compiled execution. |

The test inventory is a map to existing evidence, not a claim that every optional test ran while writing this document. For historical pass/skip counts, compiler restrictions and acceptance tolerances, read [VALIDATION.md](VALIDATION.md).

## 15. Manual-review checklist

- Match the source revision and blob before relying on these line numbers.
- Confirm heights, map spacing, origins, foot offsets and distances all use meters.
- Distinguish height-map spacing from the requested translation-grid spacing.
- Check that the selected saved stance mask has exactly `N` true entries per phase.
- Check the rotation sign and the `phase, heading` flattening order.
- Confirm `M=N*(N-1)/2` and `i<j`: no self-pairs or duplicate reversed pairs.
- Confirm the `h00–h11` triangle diagonal and the row/column order of the height array.
- Confirm common bounds contain all saved projected feet; stance selection and valid-domain construction have different roles.
- Check that line 73 casts slopes to float32 before `atan`, while sums use float64.
- Check the position sum at line 77, pair denominator `M`, global denominator `K*L` and single final square root.
- Keep the saved geometry, finite sampling and downstream interpretation separate: this file computes geometric exposure, not contact forces or locomotion speed.

No `7.76-degree` fitted scale, quadratic speed-ranking objective or STL export appears in `core.py`. Those downstream workflows are outside this program's calculation.

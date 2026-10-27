"""Run from the cloned repository after `python -m pip install -e .`."""
from pathlib import Path
import numpy as np
import torch
from terrain_roughness import calculate_roughness

ROOT = Path(__file__).resolve().parents[1]
# Many small CPU tensor operations are faster with one thread on the tested
# machine. This example setting changes execution resources, not the metric.
torch.set_num_threads(1)
with np.load(ROOT / "data/heightmaps/libo_x1p00.npz", allow_pickle=False) as terrain:
    result = calculate_roughness(
        terrain["height"],
        wave_number=3.0,
        grid_spacing_m=(float(terrain["dx_m"]), float(terrain["dy_m"])),
        map_origin_xy_m=terrain["origin_xy_m"],
        segments=19,
    )
print(f"R_g = {result:.8f} deg")  # Approximately 9.43447229 deg.

"""Command-line interface for calibrated physical height arrays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .core import calculate_roughness


def main():
    parser = argparse.ArgumentParser(description="Gait-conditioned terrain roughness in degrees. Height values must be meters.")
    parser.add_argument("height_file", type=Path)
    parser.add_argument("--wave", type=float, required=True)
    parser.add_argument("--spacing-mm", type=float, help="Required for .npy; .npz can provide dx_m/dy_m.")
    parser.add_argument("--segments", type=int, default=19)
    parser.add_argument("--height-key", default="height")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--origin-mm", type=float, nargs=2, metavar=("X0", "Y0"))
    parser.add_argument("--bounds-mm", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--scan-step-mm", type=float, default=2.)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--compile-gpu", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    spacing = None if args.spacing_mm is None else args.spacing_mm / 1000
    origin = None if args.origin_mm is None else np.array(args.origin_mm) / 1000
    if args.height_file.suffix.lower() == ".npz":
        with np.load(args.height_file, allow_pickle=False) as data:
            height = data[args.height_key].copy()
            if spacing is None and "dx_m" in data:
                spacing = (float(data["dx_m"]), float(data["dy_m"]) if "dy_m" in data else float(data["dx_m"]))
            if origin is None and "origin_xy_m" in data:
                origin = data["origin_xy_m"].copy()
    elif args.height_file.suffix.lower() == ".npy":
        height = np.load(args.height_file, allow_pickle=False)
    else:
        parser.error("Use a .npy or .npz array of physical heights in meters.")
    if spacing is None:
        parser.error("Provide --spacing-mm or an .npz containing dx_m (and optionally dy_m).")
    value = calculate_roughness(height, args.wave, grid_spacing_m=spacing,
        segments=args.segments, device=args.device, compile_gpu=args.compile_gpu,
        scan_step_m=args.scan_step_mm/1000,
        map_origin_xy_m=(0., 0.) if origin is None else origin,
        terrain_bounds_m=None if args.bounds_mm is None else np.array(args.bounds_mm)/1000)
    print(f"Gait-conditioned terrain roughness: {value:.8f} deg")


if __name__ == "__main__":
    main()

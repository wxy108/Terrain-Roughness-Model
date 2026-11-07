# Code and algorithm guide

This guide maps the repository's code to the calculation it performs. Start here to understand the modules, follow one evaluation, or find the right script for reproduction and validation.

The public function is `calculate_roughness()`. Given a calibrated physical height map and one saved gait, it returns the **complete-graph stance-foot pair-angle RMS in degrees**. For the derivation, see [Mathematical method](METHOD.md); for installation and complete command examples, see the [README](../README.md).

## Recommended reading order

1. [Minimal Python example](../examples/basic_usage.py): see the inputs and one function call.
2. [`calculate_roughness()` in `core.py`](../src/terrain_roughness/core.py): follow the top-level calculation and parameter documentation.
3. The internal functions in the algorithm table below: follow geometry, scanning, height sampling and reduction.
4. [Mathematical method](METHOD.md): connect the implementation to the equations.
5. [Tests](../tests/) and [validation record](VALIDATION.md): understand what has been checked and the scope of those checks.

## File-to-module map

| File | Module or responsibility | Main inputs and outputs |
|---|---|---|
| [`src/terrain_roughness/__init__.py`](../src/terrain_roughness/__init__.py) | Public package interface. Exports `calculate_roughness` and the package version. | Enables `from terrain_roughness import calculate_roughness`. |
| [`src/terrain_roughness/core.py`](../src/terrain_roughness/core.py) | Core geometry and roughness algorithm, including CPU/CUDA execution and optional compilation. | Physical height map, saved wave/segment selection and scan settings → one RMS value in degrees. |
| [`src/terrain_roughness/cli.py`](../src/terrain_roughness/cli.py) | Command-line entry point. Reads `.npy`/`.npz` files, interprets arguments and converts CLI millimeters to API meters. | File path and CLI arguments → a call to the public function and a printed result. |
| [`examples/basic_usage.py`](../examples/basic_usage.py) | Minimal executable usage example. Loads the included height-×1 terrain and evaluates wave 3 with 19 leg-bearing segments. | Included terrain array → printed RMS value. |
| [`studies/reproduce_study.py`](../studies/reproduce_study.py) | Full study reproduction. Shares base-height queries across nine nonflat amplitudes, then applies the angle calculation separately at each amplitude. | Included height maps and gait bank → study CSV, per-kernel arrays, timings and optional reference comparisons. |
| [`studies/benchmark.py`](../studies/benchmark.py) | Matched GPU timing of ordinary Torch execution and `torch.compile` execution. Uses the multi-amplitude engine from `reproduce_study.py`. | Included study inputs and a working CUDA/compiler setup → timing and numerical-comparison records. |
| [`studies/plot_results.py`](../studies/plot_results.py) | Scientific figure generation from retained results. | Reference CSV and per-kernel arrays → four figures in PNG and SVG formats. |
| [`studies/verify_data.py`](../studies/verify_data.py) | Data-file integrity check. | Manifest and included arrays → SHA-256 verification or an explicit mismatch error. |
| [`tests/test_metric.py`](../tests/test_metric.py) | Mathematical and numerical checks. Tests analytic planes, constant-height ground, vertical offsets, chunk sizes, triangle interpolation, the final grid vertex and a CUDA plane case when available. | Constructed surfaces and saved geometry → assertions against known values or invariants. |
| [`tests/test_geometry_and_inputs.py`](../tests/test_geometry_and_inputs.py) | Geometry-bank, scan-domain and input-validation checks. | All 30 saved wave/segment combinations and valid/invalid inputs → shape, distance, boundary and error-handling assertions. |
| [`tests/test_data_and_regression.py`](../tests/test_data_and_regression.py) | Data hashes, all retained energy aggregates, three optional full-terrain regressions and an optional compiled regression. | Included data and reference results → integrity and numerical regression assertions. |
| [`tests/conftest.py`](../tests/conftest.py) | Shared test configuration and fixtures. Registers the slow/compiled switches and temporarily sets one CPU thread during tests. | Pytest options → enabled checks and shared test resources. |
| [`pyproject.toml`](../pyproject.toml) | Packaging and entry-point configuration. Includes the gait bank in the Python package and maps `terrain-roughness` to `cli.main`. | Package metadata → installation behavior, optional dependencies and test markers. |
| [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) | GitHub Actions CPU checks on Ubuntu and Windows using Python 3.11. | A push, pull request or manual run → installation and quick-test jobs. |

## Core algorithm: function-to-step map

The core algorithm is concentrated in **one file**, [`core.py`](../src/terrain_roughness/core.py). These functions and methods are its internal modules:

| Function or method | Algorithm step | Output and interpretation |
|---|---|---|
| `_gait_bank()` | Load and cache the packaged `gait_bank.npz`. | Saved projected foot positions, stance masks, headings and common geometry bounds. |
| `_kernels(wave_number, segments)` | Select one saved gait, retain stance feet, rotate them through all headings and enumerate unordered foot pairs. | Foot coordinates with shape `[1800, segments, 2]`, horizontal pair distances, pair indices and bank metadata. The 1,800 kernels are 50 phases × 36 headings. |
| `_scan_centers(...)` | Determine the common valid translation rectangle and build an endpoint-inclusive grid. | Translation centers with shape `[L, 2]`. On the included scan-domain terrain, `L = 36,300 = 220 × 165`. |
| `_PairEnergy._slopes(...)` | Translate each foot onto the height map, sample heights using the prescribed triangle interpolation and divide absolute height differences by horizontal pair distances. | Pair slopes with shape `[kernel batch, position batch, pair]`. |
| `_PairEnergy.forward(...)` | Apply `atan`, square the pair angles and reduce over pairs and the current positions. | One partial energy sum per kernel. It is divided by the number of pairs, but still summed over the current translation points. |
| `calculate_roughness(...)` | Validate inputs; select CPU/CUDA; organize memory chunks; accumulate energy; normalize over all kernels and positions; take the final square root and convert radians to degrees. | One Python `float`: the final gait-conditioned angle RMS. |

The public interface is available through `__init__.py`; names beginning with `_` are internal implementation details.

## Follow one evaluation

```text
Python API: calibrated height_map + wave_number + grid_spacing_m
    |
    v
calculate_roughness()
    +-- validate physical inputs and scan/resource settings
    +-- _kernels() -> _gait_bank() -> stance geometry and foot pairs
    +-- _scan_centers() -> shared valid translation grid
    +-- _PairEnergy, evaluated in kernel and position chunks
    |      +-- _slopes(): sample terrain heights and form pair slopes
    |      +-- forward(): atan -> square -> partial pair-energy sums
    +-- accumulate every chunk with its actual sample count
    +-- divide by total kernels and translation points
    +-- square root -> degrees -> return float
```

The command-line route adds file reading and argument conversion before the same public call:

```text
terrain-roughness -> cli.main() -> calculate_roughness()
```

For each unordered stance-foot pair, the calculation uses:

```text
beta = atan(abs(terrain_height_i - terrain_height_j) / horizontal_distance_ij)
R_g  = degrees(sqrt(mean(beta**2 over pairs, translations, phases and headings)))
```

The square root is applied **after all averaging**. Averaging local RMS values would produce a different statistic. Memory chunks do not change sample weights; the final shorter chunk contributes its actual number of positions.

## Data modules

| Location | Role |
|---|---|
| [`src/terrain_roughness/gait_bank.npz`](../src/terrain_roughness/gait_bank.npz) | Saved kinematic geometry for six wave numbers and five leg-bearing segment counts, including calibrated amplitudes. Runtime calculation selects these saved data rather than running a locomotion simulation. |
| [`data/heightmaps/`](../data/heightmaps/) | Ten actual scan-domain height arrays, including the flat control. |
| [`data/source/`](../data/source/) | The full parent base heightfield used to prepare the scan-domain arrays. |
| [`data/manifest.json`](../data/manifest.json) | Coordinates, units, provenance and file hashes. |
| [`results/reference_50x36.csv`](../results/reference_50x36.csv) | The 300 retained condition-level reference values: 6 waves × 5 segment counts × 10 height multipliers. |
| [`results/kernel_energy/`](../results/kernel_energy/) | Retained mean-square angle arrays with shape `[1800, 9]` per wave/segment combination. Columns correspond to the nine nonflat amplitudes; flat ground is analytically zero. |
| [`results/protocol.json`](../results/protocol.json) | The retained sampling protocol and geometry context. |
| [`environment/`](../environment/) | Tested installation recipes, optional compiler requirements and the original environment inventory. |

All Python API heights, horizontal spacings and coordinates use **meters**. The height array is `[ny, nx]`: rows increase along +y and columns along +x. CLI options ending in `-mm` use millimeters, while the height array itself still uses meters. The returned RMS uses **degrees**.

The included terrain arrays and reference results belong to the study described in the README and manifest. Supplying another height map changes the evaluated terrain; it does not automatically reproduce another study's result.

## Reproduction and acceleration modules

`studies/reproduce_study.py` reuses the core geometry and interpolation instead of defining a second roughness metric:

- `load_inputs()` loads the included terrain, coordinates, common centers and manifest.
- `verify_scaling()` checks the saved height-amplitude relationship.
- `tensors_for_case()` prepares the selected gait and pair distances on the chosen device.
- `MultiScaleEnergy`, a subclass of `_PairEnergy`, shares the base pair slopes across amplitudes and evaluates `atan(scale * slope)` for each amplitude separately.
- `scan()` evaluates the 1,800 kernels in batches.
- `main()` organizes the parameter sweep, writes results and optionally compares them with retained reference values.

Scaling the final RMS directly by a height multiplier would not reproduce this nonlinear angle calculation. The single-terrain API and the joint-amplitude reproduction engine also have different batching and timing scopes, so their runtimes should be compared under matched conditions.

Ordinary CUDA execution is the default GPU path. Optional `torch.compile` acceleration requires a working compiler setup and adds first-use compilation cost. See the [README](../README.md#gpu-compilation-and-timing) and [validation record](VALIDATION.md#compiler-availability) for the tested scope.

## Which command should I run?

Run these from the repository root after installation:

| Task | Command |
|---|---|
| Quick CPU installation check; exact flat result | `terrain-roughness data/heightmaps/libo_x0p00.npz --wave 3 --device cpu` |
| One complete included example | `python examples/basic_usage.py` |
| Check included array hashes | `python studies/verify_data.py` |
| Quick analytic, geometry and retained-data tests | `python -m pytest -q` |
| Include three full real-terrain regressions | `python -m pytest -q --run-slow` |
| Recompute all 300 conditions and compare with references | `python studies/reproduce_study.py --device cuda --verify --output runs/reproduction` |
| Recompute only wave 3, 19 segments across terrain amplitudes | `python studies/reproduce_study.py --device cuda --waves 3 --segments 19 --verify` |
| Regenerate the four scientific figures; requires the plotting dependencies | `python studies/plot_results.py --output runs/figures` |
| Compare ordinary and compiled GPU execution; requires a working compiler | `python studies/benchmark.py --repeats 3 --output runs/benchmark.json` |

The complete nonflat scan is computationally substantial. The documented publication validation used CUDA for the complete study; the bounded CPU screening did not establish a complete nonflat CPU result. Use the flat command for a quick installation check. Historical validation outcomes are recorded in `docs/VALIDATION.md`; this guide does not report a new execution of those experiments.

## Where to make a change

- **Use a different calibrated terrain:** change the array passed to the API, or the CLI input file; provide its physical spacing and origin.
- **Select a saved gait:** change `wave_number` and `segments`. Unsupported wave numbers are rejected rather than interpolated.
- **Understand or change the metric:** start with `_PairEnergy` and the final reduction in `calculate_roughness()`, then use the corresponding analytic and regression tests.
- **Understand or change spatial coverage:** inspect `_scan_centers()` and `terrain_bounds_m`. A new region or sampling grid changes the averaging protocol.
- **Adjust execution resources:** use `device`, `kernel_batch`, `position_batch` and optional `compile_gpu`; these control execution rather than the intended metric.
- **Change scientific presentation:** edit `studies/plot_results.py`; it reads retained values rather than recomputing the geometry.

This repository implements the geometric roughness calculation. Downstream speed-ranking models such as `J = A + (R / 7.76 deg)**2`, their fitted scales, and printable terrain STL generation are separate workflows and are not implemented by these modules.

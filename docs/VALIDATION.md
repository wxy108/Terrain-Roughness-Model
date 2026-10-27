# Validation record

## Publication environment: 2026-10-27

- A new Python 3.11 virtual environment installed the pinned CPU recipe, without inheriting the original research environment.
- Editable installation: **54 passed, 5 skipped**. The skipped checks require CUDA, full real-terrain scanning or an optional compiler.
- A built wheel was installed into that environment in place of the editable package. The same quick suite gave **54 passed, 5 skipped**. The wheel included the gait-bank asset and imported from the environment's installed packages.
- Existing CUDA 12.8 / Torch 2.7.0 environment, including all three full real-terrain regression cases: **58 passed, 1 skipped**. The remaining optional compiler check was not enabled.
- Both the documented Python example and the CLI on CUDA returned **9.43447229°** for the included wave-3, 19-segment, height-×1 terrain. The clean CPU wheel's flat-control CLI returned exactly zero.
- The data verifier checked the SHA-256 hashes of all ten scan-domain height maps, the full parent base heightfield and the packaged gait bank.
- All retained per-kernel arrays reaggregated to all 300 original reference values.
- The four figure-generation paths ran successfully and their images were visually reviewed. Original scientific figures and editable vector versions are included separately.

The structured publication record is in [results/publication_validation.json](../results/publication_validation.json). The complete re-evaluation output is [results/reproduced_2026-10-27.csv](../results/reproduced_2026-10-27.csv).

The full nonflat example in the clean CPU environment exceeded the bounded 180-second screening limit with the default thread count and again with one CPU thread. No full CPU real-terrain result is claimed from these attempts. The example retains one CPU thread as an explicit resource setting. The README offers the exact flat control as a quick CPU installation check and recommends CUDA for the complete scan.

## Compiler availability

An attempted fresh compiled run was stopped because Windows Application Control blocked loading the newly generated `cuda_utils` module. The publication re-evaluation uses ordinary Torch CUDA and retains the same sampling and formula. No system policy was changed or bypassed.

Compilation remains optional. The earlier standalone/API and GPU benchmark validation records are retained with their original timing scope. This publication does not claim that the new repository's optional compiled study or benchmark was executed successfully under the blocked system policy.

## Numerical checks

Independent analytic tests cover a tilted plane, flat ground, vertical offset invariance, unequal map spacings, different memory chunk sizes, triangle interpolation and the final boundary vertex. The saved geometry bank is checked at every supported wave/segment combination.

The release acceptance tolerance is **0.00001°** for the analytic and complete-scan comparisons. A tolerance is an implementation acceptance criterion, not an estimate of biological or model uncertainty. Sampling remains the specified finite 50-phase, 36-heading, ≤2 mm protocol; convergence to a continuous average is not claimed.

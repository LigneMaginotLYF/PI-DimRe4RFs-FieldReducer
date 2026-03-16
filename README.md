# PI-DimRe4RFs-FieldReducer

## Overview

This repository implements a dimension-reduction surrogate modeling framework for material fields in a Biot-equation consolidation setting. The goal is to learn a mapping **M**: **ξ_E** → **ξ'** from high-dimensional KL expansion coefficients to a reduced parameter vector, such that a surrogate **S**(**ξ'**) ≈ **Y** (settlement profile) recovers the same physical response as the original complex fields.

### Four-Phase Workflow

| Phase | Description |
|-------|-------------|
| 1 | Generate 500 training samples with Matérn KL-expanded Young's modulus fields and run 2D Biot solver |
| 2 | Build a reduced LUT (default: 2000 grid points in ξ'-space), precompute responses, fit surrogate **S**: ξ' → Y |
| 3 | Train dimension reducer **M**: ξ_E → ξ' using frozen **S** and output-space MSE loss |
| 4 | Evaluate on test split; compute R², RMSE, relative L² error; produce plots |

---

## Key Features

### Phase 1 – Fixed DCT basis with Matérn-shaped sampling and DC-encoded mean

When `random_field.field_basis: "dct"`, Phase 1 uses a **fixed 2D DCT-II basis** for log(E).  The coefficient vector ξ_E has consistent semantics across all samples because the basis never changes.  Matérn-like spatial structure is achieved by shaping the *variance* of each DCT coefficient using an approximate 2D Matérn spectral density (controlled by `nu_sampling`/`length_scale_sampling`).

**Per-sample mean variation** (`E_ref_sampling: true`, DCT only): each sample draws a scaling factor `f ~ Uniform(E_ref_factor_range)` and encodes the mean shift directly into the DC DCT coefficient (index 0, mode (0,0)).  This means the ξ_E vector alone carries the mean information—no extra input feature is required.

### Phase 2 – Smooth DCT surrogate output

Set `surrogate.output_representation: "dct"` (and `n_output_modes: 8`) to train the Phase-2 surrogate to predict **DCT coefficients** of the settlement profile Y(x) rather than raw node values.  At inference, `ReducedLUT.predict()` applies the inverse DCT to return the full profile.  The truncated DCT basis constrains predictions to smooth (band-limited) functions, preventing the oscillatory artefacts that can appear with direct node-by-node prediction.

A **roughness diagnostic** (mean ‖ΔY‖) is logged at Phase-2 evaluation alongside R² and RMSE to make any remaining oscillations visible.

### Phase 4 – Direct-physics qualitative plots

When `phase4.use_direct_physics_for_plots: true` (default), the 5-sample settlement comparison plots use the **direct Biot solver** on the reconstructed reduced E field rather than the Phase-2 surrogate.  This isolates dimension-reducer + reduced-field error in the plots, removing Phase-2 surrogate artefacts from the visual comparison.  Quantitative metrics (R², RMSE, etc.) still use the surrogate for speed.

### Plot improvements

Settlement comparison plot y-axes always have `ylim(bottom=0.0)` for absolute-scale context.  The x-axis uses the full solver grid (`n_nodes_x` points); collocation positions are overlaid as markers only.

---

## Environment Setup

```bash
git clone https://github.com/LigneMaginotLYF/PI-DimRe4RFs-FieldReducer
cd PI-DimRe4RFs-FieldReducer
pip install -r requirements.txt
```

---

## Run Command

```bash
python train.py --preset presets/stage1_d1_polynomial.yaml
```

Additional CLI options:

```
--config PATH          Base config file (default: config.yaml)
--preset PATH          Preset YAML to override config values
--output-dir DIR       Results output directory (default: results/<timestamp>)
--phases 1,2,3,4       Comma-separated phases to run (default: all four)
--n-samples N          Override number of training samples
--surrogate-type TYPE  Override surrogate type: nn or pce
```

### Fast / Smoke Run

To verify the pipeline quickly with small settings:

```bash
python train.py --preset presets/stage1_d1_polynomial.yaml \
    --n-samples 20 --surrogate-type pce --output-dir results/smoke
```

---

## Artifact Structure

After a full run, the following files are created:

```
data/
  X_train.npy          (n_samples, 5)   KL coefficients for Young's modulus
  Y_train.npy          (n_samples, n_x) Settlement profiles from Phase 1
  lut_grid_points.npy  (n_grid, 3)     Reduced parameter grid (default n_grid=2000)
  lut_responses.npy    (n_grid, n_x)   LUT responses

models/
  reduced_lut/
    surrogate_nn.pt        NN surrogate S weights
    surrogate_nn_full.pt   NN surrogate S full model
    surrogate_pce.pkl      PCE surrogate S
    grid_points.npy        LUT grid copy
    responses.npy          LUT responses copy
    config.json            Surrogate metadata (type, R², date, dims)
  dimension_reducer_nn.pt  NN dimension reducer M
  dimension_reducer_pce.pkl  PCE dimension reducer M

plots/
  material_fields/         Original vs reduced E field comparisons
  settlement_comparison/   Original vs predicted settlement profiles
  sensitivity/             Sobol first-order sensitivity indices
  aggregate/               R², RMSE, relative L² distributions

results/<run_id>/
  metrics.json             Aggregate test metrics
  run_summary.txt          Config, timings, metrics, artifact paths
```

**Note:** `data/`, `models/`, and `plots/` directories may accumulate artifacts across runs. Phase 2 **always recomputes** the LUT and retrains the surrogate on every run (cache reuse is intentionally disabled to prevent stale-artifact contamination when the problem setup changes). Set `dataset.reuse: true` in `config.yaml` to reuse an existing Phase-1 dataset.

---

## Testing

```bash
python -m pytest tests/test_basic.py -v
```

Tests cover:
- Config loading and deep merge
- Preset override merge
- Scientific notation coercion
- Matérn kernel positive semi-definiteness
- KL field generation (positive E values, no NaN)
- DCT field generation: fixed basis, mean encoding via DC coefficient, E_ref_sampling
- Phase-2 DCT surrogate output: correct reconstruction shape, smoother-than-direct output
- Biot solver output shape and physical consistency
- NN and PCE surrogate fit/predict
- Validation metric computation
- Full pipeline smoke run (PCE and NN)
- Phase 2 surrogate reuse/cache
- Settlement comparison plotting: correct y-axis lower bound, no crash when collocation length ≠ n_nodes_x

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: yaml` | Run `pip install pyyaml` |
| Out-of-memory on full run | Reduce `reduced_lut.n_grid_points` or `solver.n_nodes_x/z` in `config.yaml` |
| Phase 2 recomputing despite existing artifacts | Delete `models/reduced_lut/` to force a clean rebuild |
| NaN in settlement | Check that `E_ref` and KL coefficient range are physically reasonable |

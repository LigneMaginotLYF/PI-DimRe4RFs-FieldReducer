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

**Note:** `data/`, `models/`, and `plots/` are shared across runs for reuse. Phase 2 uses cached surrogate artifacts automatically if `models/reduced_lut/config.json` exists.

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
- Biot solver output shape and physical consistency
- NN and PCE surrogate fit/predict
- Validation metric computation
- Full pipeline smoke run (PCE and NN)
- Phase 2 surrogate reuse/cache

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: yaml` | Run `pip install pyyaml` |
| Out-of-memory on full run | Reduce `reduced_lut.n_grid_points` or `solver.n_nodes_x/z` in `config.yaml` |
| Phase 2 recomputing despite existing artifacts | Delete `models/reduced_lut/` to force a clean rebuild |
| NaN in settlement | Check that `E_ref` and KL coefficient range are physically reasonable |

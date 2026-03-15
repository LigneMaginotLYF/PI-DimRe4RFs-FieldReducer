# Configuration Parameter Guide

This document describes every tunable parameter available in `config.yaml` (and overridable via preset files in `presets/`).

---

## `dataset`

Controls training-data generation (Phase 1).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 500 | Number of (KL-field, settlement) pairs to generate. More samples improve reducer generalisation but increase Phase-1 cost linearly. |
| `n_kl_terms_E` | int | 5 | Number of KL expansion terms for the Young's modulus field. Controls input dimensionality of the reducer (M: ξ_E ∈ ℝ^n_kl → ξ' ∈ ℝ³). Larger values capture more field variability. |
| `seed` | int | 42 | Global random seed for reproducibility. |
| `reuse` | bool | false | When `true`, loads `data/X_train.npy` and `data/Y_train.npy` instead of regenerating. Enables resuming without rerunning the expensive Biot solver. |
| `path_X` | str | `data/X_train.npy` | Custom load/save path for the KL-coefficient array. Effective only when `reuse: true` or to redirect saves. |
| `path_Y` | str | `data/Y_train.npy` | Custom load/save path for the settlement-profile array. |

**Interactions:** `reuse: true` requires matching `n_kl_terms_E` and `n_samples` from the saved run.

---

## `material`

Physical material constants used by the Biot solver.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `E_ref` | Pa | 1.0e7 | Reference Young's modulus. Sets the baseline stiffness; KL expansion modulates around `E_ref`. |
| `permeability_ref` | m² | 1.0e-12 | Reference isotropic permeability used to scale the reduced log-permeability parameters ξ'₁, ξ'₂. |
| `permeability_h` | m² | 1.0e-12 | Horizontal permeability used in Phase-1 data generation (heterogeneous fields). |
| `permeability_v` | m² | 1.0e-12 | Vertical permeability for Phase-1. |
| `poisson_ratio` | — | 0.3 | Drained Poisson's ratio (0 < ν < 0.5). |
| `biot_coefficient` | — | 0.8 | Biot-Willis coefficient (0 ≤ b ≤ 1). |
| `fluid_viscosity` | Pa·s | 1.0e-3 | Pore-fluid dynamic viscosity. |
| `porosity` | — | 0.3 | Total porosity (affects fluid storage). |
| `fluid_bulk_modulus` | Pa | 2.2e9 | Bulk modulus of the pore fluid (water ≈ 2.2 GPa). |
| `applied_load` | Pa | 1.0e6 | Vertical surface load applied to the top boundary. |
| `pore_pressure_bottom` | Pa | 1.0e5 | Fixed pore pressure at the bottom boundary (drainage condition). |

**Compute cost:** These parameters only affect the Biot solver, hence Phase 1 and Phase 2 cost. They do not influence Phase 3/4 training directly.

---

## `domain`

Physical dimensions of the 2-D consolidation domain.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `length_x` | m | 1.0 | Width of the domain. |
| `length_z` | m | 1.0 | Depth of the domain. |

---

## `solver`

Finite-difference solver settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | `"2d"` | Solver formulation (only `"2d"` supported). |
| `response_mode` | str | `"steady_state"` | Type of Biot solution (only `"steady_state"` supported). |
| `n_nodes_x` | int | 20 | Number of FD grid nodes in the x direction. Output settlement vector has this length. Higher values give smoother fields but increase FD solve time and output dimensionality. |
| `n_nodes_z` | int | 20 | Number of FD grid nodes in the z direction. Affects integration accuracy for vertical settlement. |

**Accuracy vs. cost:** Doubling `n_nodes_x` and `n_nodes_z` roughly quadruples FD cost and doubles the surrogate output dimension, increasing Phase-2 and Phase-3 training cost.

---

## `random_field`

Parameters for sampling Matérn KL random fields in Phase 1.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `covariance` | str | `"matern"` | Covariance kernel (only `"matern"` implemented). |
| `nu_sampling` | bool | true | If true, sample the Matérn smoothness ν from `nu_range` for each sample. |
| `nu_range` | [float, float] | [0.5, 2.5] | Uniform range for ν. Higher ν → smoother fields. |
| `length_scale_sampling` | bool | true | If true, sample the length scale from `length_scale_range`. |
| `length_scale_range` | [float, float] | [0.1, 0.5] | Uniform range for the correlation length scale (relative to domain size). Larger values → more spatially correlated fields. |

---

## `dimension_reducer`

Settings for the dimension reducer M (Phase 3).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d` | int | 1 | Target reduced dimension (informational; actual output is always 3 for [E', k'_h, k'_v]). |
| `basis_type` | str | `"polynomial"` | Basis type for the PCE reducer (informational). |
| `basis_order` | int | 1 | Polynomial degree for PCE-based reducer. |
| `types` | list[str] | *(from surrogate.type)* | List of reducer types to train simultaneously, e.g. `["nn", "pce"]`. When omitted, uses `surrogate.type`. Enables comparison plots. |

---

## `reduced_lut`

Settings for the reduced look-up table (LUT) built in Phase 2.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_grid_points` | int | 2000 | Number of points in the reduced parameter space grid. More points → better surrogate coverage but longer precomputation. |
| `grid_type` | str | `"random"` | Grid sampling strategy: `"random"` (standard normal) or `"structured"` (regular grid). |
| `reuse` | bool | false | When `true` and `models/reduced_lut/config.json` exists, skip LUT precomputation and surrogate training. Use when iterating on downstream phases. |
| `path` | str | `models/reduced_lut` | *(unused by default; reserved for future custom paths)* |

**Interactions with reuse:** `reuse: true` is equivalent to `force_recompute=False` when a cached surrogate exists. Set to `false` to always retrain (e.g., after changing `n_grid_points` or material parameters).

---

## `surrogate`

Settings for the Phase-2 surrogate S: ξ' → Y and the Phase-3 reducer backbone.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | `"nn"` | Primary surrogate type: `"nn"` (neural network) or `"pce"` (polynomial chaos). |
| `types` | list[str] | *(from type)* | Train multiple surrogate types simultaneously, e.g. `["nn", "pce"]`. Produces comparison metrics and plots. |
| `hidden_dim` | int | 64 | Hidden layer width for the NN surrogate. Increase for more expressive surrogates; diminishing returns beyond ~128. |
| `n_blocks` | int | 3 | Number of residual blocks in the NN. Deeper networks can fit more complex mappings. |
| `epochs` | int | 200 | Training epochs. Increase if validation loss is still decreasing. |
| `learning_rate` | float | 1.0e-3 | Initial Adam learning rate. A cosine or plateau scheduler reduces this automatically. |
| `batch_size` | int | 64 | Mini-batch size. Smaller batches → noisier gradients (sometimes beneficial). |
| `basis_order` | int | 3 | Polynomial degree for PCE surrogate. Degree 3 with 3 inputs gives 20 basis functions; higher degrees are expensive. |

**Accuracy vs. cost:**
- `epochs` has a near-linear effect on training time.
- `hidden_dim` and `n_blocks` affect model capacity and inference cost quadratically/linearly.
- PCE with `basis_order=3` and 3 inputs ≈ 20 coefficients; well-suited for smooth surrogates with small LUTs.

---

## `validation`

Train/validation/test split fractions (must sum to ≤ 1.0).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_fraction` | float | 0.6 | Fraction of Phase-1 data used for reducer training. |
| `val_fraction` | float | 0.2 | Fraction used for validation during training (early stopping / LR scheduling). |
| `test_fraction` | float | 0.2 | Fraction held out for Phase-4 test evaluation. |

**Note:** `train_fraction + val_fraction + test_fraction` should equal 1.0. Any remainder is silently assigned to the test set.

---

## Reuse flags — interaction summary

```
dataset.reuse=true      →  Phase 1 is skipped; arrays loaded from data/
reduced_lut.reuse=true  →  Phase 2 is skipped; surrogate loaded from models/reduced_lut/
```

Default save paths match default load paths, so no additional path overrides are needed unless you redirect to a custom location.

---

## Simultaneous NN + PCE training

To train both surrogate types and obtain comparison metrics:

```yaml
surrogate:
  types: ["nn", "pce"]
  epochs: 200
  basis_order: 3

dimension_reducer:
  types: ["nn", "pce"]
```

- Phase 2 trains `surrogate_nn.pt` and `surrogate_pce.pkl`; produces `results/<run>/plots/aggregate/surrogate_comparison.png`.
- Phase 3 trains `dimension_reducer_nn.pt` and `dimension_reducer_pce.pkl`.
- Phase 4 evaluates both reducers and produces a comparison plot.
- Separate Phase-2 evaluation folders: `models/reduced_lut/nn/evaluation/` and `models/reduced_lut/pce/evaluation/`.

---

## Artifact directory layout (per run)

```
results/<run_id>/
├── metrics.json          # Phase-4 test R², RMSE, relL2
├── run_summary.txt       # Config, timings, artifact paths, R² diagnosis
└── plots/
    ├── material_fields/
    │   └── comparison.png    # Original E vs reduced E' + diff map
    ├── settlement_comparison/
    │   └── comparison.png    # Profile overlays for 5 test samples
    ├── aggregate/
    │   ├── metrics.png       # R²/RMSE/relL2 distributions
    │   └── surrogate_comparison.png  # (only with multiple types)
    └── sensitivity/
        └── sobol_indices.png # First-order Sobol sensitivity of M

models/
├── reduced_lut/
│   ├── config.json           # Surrogate metadata (type, R², date)
│   ├── grid_points.npy
│   ├── responses.npy
│   ├── surrogate_nn.pt / surrogate_pce.pkl
│   ├── nn/evaluation/
│   │   ├── metrics.json      # Phase-2 independent test-set metrics
│   │   └── settlement_comparison/comparison.png
│   └── pce/evaluation/
│       └── ...
├── dimension_reducer_nn.pt
└── dimension_reducer_pce.pkl

data/
├── X_train.npy
├── Y_train.npy
├── X_test.npy
├── Y_test.npy
├── lut_grid_points.npy
└── lut_responses.npy
```

# Configuration Parameter Guide

This document describes every tunable parameter available in `config.yaml`
(and overridable via preset files in `presets/`).

---

## `dataset`

Controls training-data generation (Phase 1).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | int | 500 | Number of (field, settlement) pairs to generate. |
| `n_terms_E` | int | 5 | Number of field expansion terms for the Young's modulus field. Controls input dimensionality of the reducer (M: ξ_E ∈ ℝ^n_terms → ξ' ∈ ℝ^d). **Preferred key.** |
| `n_kl_terms_E` | int | *(alias)* | Legacy alias for `n_terms_E`. Still accepted but emits a `DeprecationWarning`; prefer `n_terms_E` for new configs. |
| `seed` | int | 42 | Global random seed for reproducibility. |
| `reuse` | bool | false | When `true`, loads `data/X_train.npy` / `data/Y_train.npy` instead of regenerating. |
| `path_X` | str | `data/X_train.npy` | Custom load/save path for the coefficient array (effective only when `reuse: true`). |
| `path_Y` | str | `data/Y_train.npy` | Custom load/save path for the settlement-profile array. |

**Interactions:** `reuse: true` requires that `n_terms_E` matches the saved run.

---

## `material`

Physical material constants used by the Biot solver.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `E_ref` | Pa | 1.0e7 | Reference Young's modulus. |
| `permeability_ref` | m² | 1.0e-12 | Reference isotropic permeability. |
| `permeability_h` | m² | 1.0e-12 | Horizontal permeability (fixed; only E is treated as a random field). |
| `permeability_v` | m² | 1.0e-12 | Vertical permeability (fixed). |
| `poisson_ratio` | — | 0.3 | Drained Poisson's ratio (0 < ν < 0.5). |
| `biot_coefficient` | — | 0.8 | Biot-Willis coefficient (0 ≤ b ≤ 1). |
| `fluid_viscosity` | Pa·s | 1.0e-3 | Pore-fluid dynamic viscosity. |
| `porosity` | — | 0.3 | Total porosity. |
| `fluid_bulk_modulus` | Pa | 2.2e9 | Bulk modulus of the pore fluid. |
| `applied_load` | Pa | 1.0e6 | Vertical surface load at the top boundary. |
| `pore_pressure_bottom` | Pa | 1.0e5 | Fixed pore pressure at the bottom (drainage) boundary. |

---

## `domain`

Physical dimensions of the consolidation domain.

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `length_x` | m | 1.0 | Width of the domain (2-D) or ignored (1-D). |
| `length_z` | m | 1.0 | Depth of the domain. |

---

## `solver`

Finite-difference solver settings.

| Parameter | Type | Default | Allowed | Description |
|-----------|------|---------|---------|-------------|
| `type` | str | `"2d"` | `"1d"`, `"2d"` | Problem dimensionality. `"1d"` = single vertical column (`n_nodes_x` must be 1). `"2d"` = full plane-strain domain. |
| `response_mode` | str | `"steady_state"` | `"steady_state"`, `"transient"` | Physics regime. `"steady_state"` uses an analytical pressure profile. `"transient"` performs explicit time-stepping up to `t_final`. |
| `n_nodes_x` | int | 20 | ≥ 1 | Number of FD grid nodes in x. For 1-D problems set to 1. |
| `n_nodes_z` | int | 20 | ≥ 2 | Number of FD grid nodes in z (vertical). |
| `t_final` | float | 1.0 | > 0 | Non-dimensional final consolidation time T = c_v·t/L_z². Only used when `response_mode = "transient"`. T = 1.0 ≈ 95 % primary consolidation. |
| `max_time_steps` | int | 2000 | ≥ 1 | Cap on the number of explicit time steps for transient simulations. The solver chooses the stability-limited step count up to this cap and emits a `UserWarning` if capped. Increase for production runs requiring higher physical accuracy. |

**Invalid combinations:**
- `type: "1d"` with `n_nodes_x > 1` → validation error.
- `response_mode: "transient"` without `t_final` → validation error.

---

## `random_field`

Parameters for sampling random material fields in Phase 1.

| Parameter | Type | Default | Allowed | Description |
|-----------|------|---------|---------|-------------|
| `covariance` | str | `"matern"` | `"matern"` | Covariance kernel family. Only `"matern"` is currently implemented. |
| `field_basis` | str | `"kl"` | `"kl"`, `"dct"` | **Basis used for Phase-1 field generation.** `"kl"` (default): per-sample Matérn-KL eigenbasis; recomputed for each (ν, ℓ) draw. `"dct"`: fixed 2D DCT-II basis; coefficient variance is shaped by an approximate 2D Matérn spectral density without changing the basis. See *DCT basis notes* below. |
| `logE_std` | float | 1.0 | > 0 | Global amplitude multiplier on log(E). Decreasing this value makes GT settlements smoother / less variable between samples. Alias `field_fluctuation_scale` is also accepted; `logE_std` takes priority. |
| `field_fluctuation_scale` | float | *(alias)* | > 0 | Alias for `logE_std`. Deprecated in favour of `logE_std`. |
| `nu_sampling` | bool | true | — | If `true`, the Matérn smoothness ν is drawn uniformly from `nu_range` for each sample. Set `false` for fixed ν (required for identity-mode verification). For DCT basis, this reshapes coefficient variance without changing basis functions. |
| `nu_range` | [float, float] | [0.5, 2.5] | — | Uniform range for ν. Higher ν → smoother fields. |
| `nu_ref` | float | 1.5 | — | Fixed ν used when `nu_sampling: false`. Also used as the KL-basis reference when `basis_type: "kl"`. |
| `length_scale_sampling` | bool | true | — | If `true`, length scale ℓ is drawn from `length_scale_range` per sample. |
| `length_scale_range` | [float, float] | [0.1, 0.5] | — | Uniform range for the correlation length scale. |
| `length_scale_ref` | float | 0.3 | — | Fixed ℓ used when `length_scale_sampling: false`. Also used as the KL-basis reference when `basis_type: "kl"`. |

### DCT basis notes

When `field_basis: "dct"`:
- The 2D DCT-II basis modes are ordered by frequency magnitude (smoothest first).
  The first `n_terms_E` modes are used.
- The basis is **identical across all samples and all (ν, ℓ) draws**.
  Two samples that happen to share the same coefficient vector `ξ_E` will
  produce the *same* log-field regardless of which (ν, ℓ) was used to draw them.
- Matérn-like spatial structure is achieved by sampling each coefficient
  `a_k ~ N(0, σ_k²)` where `σ_k ∝ (2ν/ℓ² + ‖ω_k‖²)^{-(ν+1)/2}` (approximate
  2D Matérn spectral density; documented in `src/field_generator.py` (`DCTField` class)).
- The coefficient vector stored in `X_train` includes the Matérn-shaped variance,
  so the mapping M: ξ_E → ξ' is trained on these scaled coefficients.
- Set `dimension_reducer.basis_type: "dct"` in Phase 2 to use the same DCT
  modes for LUT reconstruction (basis consistency).

**For identity-mode verification**, set both `nu_sampling: false` and `length_scale_sampling: false` so that Phase-1 generation and Phase-2 reconstruction share the same KL/DCT basis parameters (→ machine-precision equivalence).

---

## `dimension_reducer`

Settings for the dimension reducer M (Phase 3).

| Parameter | Type | Default | Allowed | Description |
|-----------|------|---------|---------|-------------|
| `d` | int | 1 | 1 … n_terms | Reduced-space dimension. Number of scalar parameters that describe the low-dimensional material field. |
| `basis_type` | str | `"polynomial"` | `"polynomial"`, `"kl"`, `"dct"` | Basis used to reconstruct the material field from the reduced coefficients ξ'. `"polynomial"` = multivariate polynomial in (x/L_x, z/L_z). `"kl"` = truncated KL expansion with reference Matérn parameters. `"dct"` = first d 2D DCT-II modes consistent with Phase-1 when `random_field.field_basis = "dct"`. |
| `basis_order` | int | 1 | 1 … 10 | Highest polynomial degree (only for `basis_type = "polynomial"`). Controls the number of available basis functions: order 1 → 3, order 2 → 6, order 3 → 10. `d` must not exceed this count. |
| `mode` | str | `"learned"` | `"learned"`, `"identity"` | Mapping model mode. `"learned"` trains a NN or PCE. `"identity"` bypasses learning and routes ξ_E directly as ξ' (first d components); requires `d = n_terms_E`. |
| `types` | list[str] | *(from surrogate.type)* | `["nn"]`, `["pce"]`, `["nn","pce"]` | Train multiple reducer types simultaneously. |

**Invalid combinations:**
- `basis_type: "polynomial"` and `d > n_poly_basis(basis_order)` → validation error.
- `basis_type: "kl"` or `"dct"` and `d > n_terms_E` → validation error.
- `mode: "identity"` and `d ≠ n_terms_E` → validation error.

---

## `reduced_lut`

Settings for the reduced look-up table (LUT) in Phase 2.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_grid_points` | int | 2000 | Number of points in the reduced parameter space grid. More points → better surrogate coverage but higher precomputation cost. |
| `grid_type` | str | `"random"` | Grid sampling strategy: `"random"` (standard normal) or `"structured"` (regular grid). |
| `reuse` | bool | false | **Currently has no effect.** Phase-2 LUT and surrogate are always recomputed to prevent stale-artifact contamination. The flag is retained for future re-enabling. |

**Config hash:** The LUT config is hashed over `(d, basis_type, basis_order, solver.type, response_mode, n_nodes_x, n_nodes_z, nu_ref, ls_ref)`. A hash change forces recomputation even if `reuse: true` were re-enabled.

---

## `surrogate`

Settings for the Phase-2 surrogate S: ξ' → Y and the Phase-3 reducer backbone.

| Parameter | Type | Default | Allowed | Description |
|-----------|------|---------|---------|-------------|
| `type` | str | `"nn"` | `"nn"`, `"pce"` | Primary surrogate/reducer type. |
| `types` | list[str] | *(from type)* | `["nn"]`, `["pce"]`, `["nn","pce"]` | Train multiple types simultaneously; produces comparison metrics and plots. |
| `hidden_dim` | int | 64 | — | Hidden layer width for NN. |
| `n_blocks` | int | 3 | — | Number of residual blocks in the NN. |
| `epochs` | int | 200 | — | Training epochs. |
| `learning_rate` | float | 1.0e-3 | — | Initial Adam learning rate. |
| `batch_size` | int | 64 | — | Mini-batch size. |

---

## `collocation`

Optional settings for collocation positions used in Phase-3 physics-driven training
and settlement-comparison plots.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_points` | int | *(all LUT points)* | Number of collocation points used in Phase-3. Defaults to `reduced_lut.n_grid_points`. |
| `positions` | list[float] | *(all x-nodes)* | Explicit physical x-positions (in metres) for collocation markers on settlement comparison plots. Defaults to all `n_nodes_x` positions. |

---

## `validation`

Train/validation/test split fractions (must sum to ≤ 1.0).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_fraction` | float | 0.6 | Fraction of Phase-1 data used for reducer training. |
| `val_fraction` | float | 0.2 | Fraction used for validation (early stopping). |
| `test_fraction` | float | 0.2 | Fraction held out for Phase-4 test evaluation. |

---

## Reuse flags — interaction summary

```
dataset.reuse=true      →  Phase 1 is skipped; arrays loaded from data/
reduced_lut.reuse=*     →  IGNORED; Phase 2 ALWAYS recomputes (by design)
```

Phase-2 reuse is intentionally disabled to prevent stale-artifact contamination when problem setup (dimension/regime/basis/etc.) changes.

---

## Simultaneous NN + PCE training

To train both surrogate types and obtain comparison metrics:

```yaml
surrogate:
  types: ["nn", "pce"]
  epochs: 200

dimension_reducer:
  types: ["nn", "pce"]
```

- Phase 2 trains `surrogate_nn.pt` and `surrogate_pce.pkl`; produces `plots/aggregate/surrogate_comparison.png`.
- Phase 2 additionally saves per-type accuracy plots under `plots/phase2_surrogate/`.
- Phase 3 trains `dimension_reducer_nn.pt` and `dimension_reducer_pce.pkl`.
- Phase 4 evaluates both reducers and produces a comparison plot.

---

## Identity-mapping verification

To verify physical consistency (identical settlement under no information loss):

```yaml
solver:
  type: "2d"
  response_mode: "steady_state"
  n_nodes_x: 20

random_field:
  nu_sampling: false
  nu_ref: 1.5
  length_scale_sampling: false
  length_scale_ref: 0.3

dimension_reducer:
  d: 5                 # must equal dataset.n_kl_terms_E
  basis_type: "kl"     # KL basis shares the same eigenvectors as Phase-1
  mode: "identity"
```

Phase 4 saves `<run>/identity_check.json` containing:
- `identity_check_max_abs_error`
- `identity_check_rmse`
- `identity_check_r2`

For exact machine-precision equivalence both `nu_sampling` and `length_scale_sampling` must be `false`.

---

## Settlement comparison plot logic

The `plots/settlement_comparison/comparison_with_collocation.png` plot shows:

1. **Blue solid line** — Ground-truth (GT) settlement profile.
2. **Red dashed line** — Reduced-space prediction profile.
3. **Green open circles** — GT values at collocation x-positions (marks WHERE the collocation constraints are applied, **not** independent LUT settlement profiles).

The x-axis represents physical x-positions (0 … L_x), not node indices.

---

## Artifact directory layout (per run)

```
results/<run_id>/
├── metrics.json            # Phase-4 test R², RMSE, relL2
├── identity_check.json     # (identity mode only) max_abs, RMSE, R²
├── run_summary.txt         # Config, timings, artifact paths
└── plots/
    ├── material_fields/
    │   └── comparison.png      # Original E vs reduced E' + diff map
    ├── settlement_comparison/
    │   ├── comparison.png      # GT + reduced curves
    │   └── comparison_with_collocation.png  # + collocation position markers
    ├── aggregate/
    │   ├── metrics.png         # R²/RMSE/relL2 distributions
    │   └── surrogate_comparison.png   # (multiple types only)
    ├── phase2_surrogate/
    │   ├── scatter_nn.png      # Phase-2 surrogate: predicted vs true (scatter)
    │   ├── profiles_nn.png     # Phase-2 surrogate: example settlement profiles
    │   └── ...pce...           # (if pce type also trained)
    └── sensitivity/
        └── sobol_indices.png   # First-order Sobol sensitivity of M

models/
├── reduced_lut/
│   ├── config.json             # Surrogate metadata (type, R², date, hash)
│   ├── grid_points.npy
│   ├── responses.npy
│   ├── surrogate_nn.pt / surrogate_pce.pkl
│   ├── nn/evaluation/
│   │   ├── metrics.json
│   │   └── settlement_comparison/comparison.png
│   └── pce/evaluation/ ...
├── dimension_reducer_nn.pt
├── dimension_reducer_pce.pkl
└── dimension_reducer_identity.pkl   # (identity mode)

data/
├── X_train.npy
├── Y_train.npy
├── X_test.npy
├── Y_test.npy
├── lut_grid_points.npy
└── lut_responses.npy
```


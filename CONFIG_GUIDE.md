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
| `E_ref_sampling` | bool | false | — | **Per-sample mean variation** (DCT basis only). When `true`, each sample draws a factor `f ~ Uniform(E_ref_factor_range)` and encodes it into the DC DCT coefficient so that the expected field mean equals `material.E_ref * f`. This increases the mean-level diversity in the training set without requiring `f` as a separate input feature. Has no effect when `field_basis != "dct"`. |
| `E_ref_factor_range` | [float, float] | [0.5, 1.5] | — | Uniform range for the per-sample E_ref scaling factor. Only used when `E_ref_sampling: true`. |

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

### DC coefficient mean encoding

When `E_ref_sampling: true` (DCT basis only):
- A per-sample factor `f ~ Uniform(E_ref_factor_range)` is drawn.
- The DC DCT coefficient `ξ_E[0]` (mode (0,0)) is **overwritten** to encode the
  desired mean shift: `ξ_E[0] = log(f) * √(n_pts) / σ_0`, where `σ_0` is the
  Matérn spectral standard deviation of the DC mode.
- This means the coefficient vector alone carries the mean information — no extra
  input feature (e.g. `E_ref_factor`) is added to `X_train`.
- The target is that `mean(E_field) ≈ E_ref * f` in expectation.

**For identity-mode verification**, set both `nu_sampling: false` and `length_scale_sampling: false` so that Phase-1 generation and Phase-2 reconstruction share the same KL/DCT basis parameters (→ machine-precision equivalence).

---

### `random_field.trend` — nonstationary spatial trend (Phase 1)

Optional polynomial trend added to log(E) during Phase-1 sample generation.
The trend is **separable**: independent polynomial terms in x and z with no
cross-product terms.  Coefficients are sampled per-sample from a uniform distribution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Master switch. When `false`, no trend is applied (returns zero contribution). |
| `order_x` | int | 1 | Highest polynomial power for the x direction. Recommended ≤ 2. |
| `order_z` | int | 1 | Highest polynomial power for the z direction. Recommended ≤ 2. |
| `include_const` | bool | `false` | Whether to add a constant (order-0) term. Usually `false` because the DC DCT coefficient / KL mean already controls the spatial mean. |
| `coeff_bounds_x` | list[[float,float]] | `[[-1,1],…]` | Per-power coefficient bounds for x. One `[min, max]` pair per power `1..order_x`. Defaults to `[-1.0, 1.0]` for unspecified terms. |
| `coeff_bounds_z` | list[[float,float]] | `[[-1,1],…]` | Per-power coefficient bounds for z. Same structure as `coeff_bounds_x`. |
| `coeff_bound_const` | [float,float] | `[-1.0, 1.0]` | Bounds for the constant term (only relevant when `include_const: true`). |

**Example** — linear-in-z tilt with mild x gradient:

```yaml
random_field:
  trend:
    enabled: true
    order_x: 1
    order_z: 2
    include_const: false
    coeff_bounds_x:
      - [-0.3, 0.3]   # coefficient for x^1
    coeff_bounds_z:
      - [-0.5, 0.5]   # coefficient for z^1
      - [-0.2, 0.2]   # coefficient for z^2
```

> **Note**: The trend is applied in **log-E space** before exponentiation, so it
> does not break the positivity constraint on E.  Large trend coefficients can
> create strong spatial gradients; keep |coeff| ≪ `logE_std` for moderate nonstationarity.

---

## `dimension_reducer`

Settings for the dimension reducer M (Phase 3).

| Parameter | Type | Default | Allowed | Description |
|-----------|------|---------|---------|-------------|
| `d` | int | 1 | 1 … n_terms | Reduced-space dimension. Number of scalar parameters that describe the low-dimensional material field. |
| `basis_type` | str | `"polynomial"` | `"polynomial"`, `"kl"`, `"dct"` | Basis used to reconstruct the material field from the reduced coefficients ξ'. `"polynomial"` = multivariate polynomial in (x/L_x, z/L_z). `"kl"` = truncated KL expansion with reference Matérn parameters. `"dct"` = first d 2D DCT-II modes consistent with Phase-1 when `random_field.field_basis = "dct"`. |
| `basis_order` | int | 1 | ≥ 1 | Highest polynomial degree (only for `basis_type = "polynomial"`). Any positive integer. Basis size = (order+1)(order+2)/2: order 1 → 3, order 2 → 6, order 3 → 10, order 4 → 15. `d` must not exceed this count. |
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
| `output_representation` | str | `"direct"` | `"direct"`, `"dct"`, `"poly"`, `"bspline"` | **Output representation for Phase-2 surrogate.** `"direct"` (default): surrogate predicts the full settlement profile Y(x) directly. `"dct"`: surrogate predicts the first `n_output_modes` 1-D DCT-II coefficients of Y(x); `predict()` applies the inverse DCT to return the full reconstructed profile. `"poly"`: surrogate predicts `n_output_modes` polynomial coefficients (degree = n_output_modes − 1) fitted to Y(x) via `np.polyfit`. `"bspline"`: surrogate predicts `n_output_modes` B-spline basis coefficients fitted to Y(x) with a B-spline of degree `bspline_degree`. |
| `n_output_modes` | int | 8 | — | Number of output modes/coefficients for transformed output representations. For `"dct"`: number of DCT-II modes. For `"poly"`: number of polynomial terms (= degree + 1). For `"bspline"`: number of B-spline basis functions (must be > `bspline_degree`). Has no effect when `output_representation: "direct"`. |
| `bspline_degree` | int | 3 | — | B-spline polynomial degree used when `output_representation: "bspline"`. Must satisfy `n_output_modes > bspline_degree`. Default: 3 (cubic splines). |

### Surrogate DCT output representation

When `output_representation: "dct"`:
- The settlement profile `Y(x)` of length `n_nodes_x` is transformed to DCT-II coefficients before training.
- The surrogate is trained to predict the first `n_output_modes` coefficients.
- At inference, `ReducedLUT.predict()` pads the truncated coefficient vector to length `n_nodes_x` and applies the inverse DCT to recover the full settlement profile.
- **Smoother outputs**: because the inverse DCT reconstructs from a low-order basis, the predictions are guaranteed to be smooth (band-limited), which prevents the oscillatory artefacts that can appear with direct node-by-node prediction.
- **Roughness metric**: Phase-2 evaluation logs a roughness diagnostic (`mean ||ΔY||`) for both ground-truth and predicted profiles; compare these values to detect oscillatory predictions.

> **Tip**: Start with `n_output_modes: 8` (captures the dominant settlement shape).
> Increase if validation R² is low due to missing high-frequency detail.

### Surrogate polynomial output representation

When `output_representation: "poly"`:
- The settlement profile is fitted with a polynomial of degree `n_output_modes − 1` using `np.polyfit` (least-squares).
- The surrogate predicts the polynomial coefficients (in descending power order, matching `np.polyfit` convention).
- At inference, `ReducedLUT.predict()` evaluates the polynomial at the normalised x-grid `linspace(0,1,n_nodes_x)`.
- Suitable for monotone or near-polynomial settlement profiles.

### Surrogate B-spline output representation

When `output_representation: "bspline"`:
- The settlement profile is fitted with `n_output_modes` B-spline basis functions of degree `bspline_degree` using `scipy.interpolate.make_lsq_spline` with uniformly spaced internal knots.
- The surrogate predicts the spline coefficients.
- At inference, `ReducedLUT.predict()` evaluates the B-spline using `scipy.interpolate.BSpline`.
- Provides flexible smooth approximation; cubic splines (`bspline_degree: 3`) are recommended.
- **Constraint**: `n_output_modes > bspline_degree` must hold; a `ValueError` is raised otherwise.

---

## `phase4`

Settings controlling Phase-4 evaluation and visualization behaviour.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_direct_physics_for_plots` | bool | `true` | **Use direct Biot solver for settlement-comparison plots** (5 sample curves). When `true`, the pipeline reconstructs the E field from the predicted reduced parameters ξ' and runs the full solver to compute Y_pred for each plot sample. This isolates reducer + reduced-field error in the plots, avoiding Phase-2 surrogate oscillations from contaminating the visual comparison. **Metrics** (R², RMSE, etc.) always use surrogate predictions for speed regardless of this setting. Set to `false` to use surrogate predictions for both plots and metrics. |

### Direct-physics plots (Phase 4)

When `use_direct_physics_for_plots: true` (default), the 5-sample settlement comparison plots are produced as follows:

1. `xi' = reducer.predict(X_test)` — reduced parameters for each test sample.
2. `E' = reduced_lut._reconstruct_field(xi')` — reconstruct the E field from ξ'.
3. `Y_plot = solver.run(E', k_h, k_v)` — run the Biot solver directly.

These direct-physics curves isolate the error introduced by the dimension reducer and the reduced-field approximation, independently of any Phase-2 surrogate artefacts.

---

## `collocation`

Single-knob configuration for collocation points.  The same positions are
used in **both** Phase-3 training loss evaluation and Phase-4/visualization
plot markers.  This ensures a single source of truth across the pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positions` | list[float] | *(all x-nodes)* | Physical x-coordinates (metres) of the collocation points.  Each value is mapped to the nearest node index in the solver's x-grid.  If omitted (or the section is `null`/empty), all `n_nodes_x` nodes are used. |

### How it works

1. `positions` is converted to node indices against `np.linspace(0, length_x, n_nodes_x)`.
2. The resulting **collocation indices** are saved to `models/reduced_lut/collocation_indices.npy` at the end of Phase 3.
3. **Phase-3 training** (NN and PCE reducers) computes the MSE loss only at these indices: `loss = MSE(Y_pred[:, colloc_idx], Y_true[:, colloc_idx])`.
4. **Phase-4 plots** load `collocation_indices.npy` and mark those x-positions as green circles on the settlement-comparison curves.

> **Note**: `collocation.positions` controls the _subset of x-nodes_ involved in
> training and plotting.  The full settlement curve is always plotted on a
> complete `n_nodes_x`-point x-axis; collocation positions are markers only.

### Example

```yaml
collocation:
  positions: [0.0, 0.25, 0.5, 0.75, 1.0]
```

With `n_nodes_x: 20` and `length_x: 1.0` this selects nodes 0, 5, 10, 14, 19
(nearest grid points), resulting in 5-point Phase-3 loss and 5 green circles
in the settlement comparison plots.

---

## `collocation_phase2` / `collocation_phase3`

Phase-specific collocation overrides that allow **independent collocation
configuration** for Phase-2 evaluation and Phase-3 training.  The lookup
order for each phase is:

1. Phase-specific section (`collocation_phase2` or `collocation_phase3`)
2. Legacy `collocation` section
3. Default: all `n_nodes_x` nodes

Each section supports three mutually exclusive specification methods (highest priority first):

| Key | Type | Description |
|-----|------|-------------|
| `indices` | list[int] | Explicit integer node indices (0-based).  Allows fully custom, uneven spacing.  Values are clipped to `[0, n_nodes_x-1]`. |
| `positions` | list[float] | Physical x-coordinates mapped to nearest node indices.  Convenient when you know the physical positions of your sensors/benchmarks. |
| `n_points` | int | Number of uniformly-spaced nodes chosen deterministically (includes both endpoints).  Simplest way to request a sparse set with guaranteed coverage. |

### Phase-2 collocation (`collocation_phase2`)

Used for evaluation markers in Phase-2 surrogate diagnostics.  Dense coverage
is recommended (the default is all nodes):

```yaml
collocation_phase2:
  # Omit any key to default to all nodes (densest possible)
```

### Phase-3 collocation (`collocation_phase3`)

Controls which x-nodes enter the Phase-3 reducer training loss.
Sparser collocation can improve generalisation and reduce training cost:

```yaml
# 5 uniformly-spaced nodes including endpoints [0, 5, 10, 14, 19] for n_nodes_x=20
collocation_phase3:
  n_points: 5

# Manually specified uneven indices (e.g. denser near boundaries)
# collocation_phase3:
#   indices: [0, 1, 5, 14, 18, 19]

# Physical positions (useful when matching sensor placement)
# collocation_phase3:
#   positions: [0.0, 0.2, 0.5, 0.8, 1.0]
```

> **Tip**: Keep `collocation_phase2` dense (all nodes) for full diagnostic coverage,
> and use `collocation_phase3` to constrain the Phase-3 training loss to a
> physically meaningful sparse set (e.g. simulated sensor locations).

---

## `phase2` — Phase-2 model configuration

Overrides for the Phase-2 LUT forward surrogate (ξ' → Y).
**Phase-2 and Phase-3 model selections are completely independent.**
These keys take precedence over `surrogate.type` / `surrogate.types`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surrogate_type` | str | *(from `surrogate.type`)* | Surrogate type for Phase 2. `"nn"` or `"pce"`. |
| `surrogate_types` | list[str] | *(from `surrogate.types`)* | Train multiple surrogate types simultaneously. |
| `pce.order` | int | 3 | PCE polynomial degree for Phase-2 (only when `surrogate_type = "pce"`). |

**Example** — use NN for Phase 2, PCE for Phase 3:

```yaml
phase2:
  surrogate_type: "nn"

phase3:
  reducer_type: "pce"
  pce:
    order: 4
```

---

## `phase3` — Phase-3 model configuration

Overrides for the Phase-3 dimension reducer (ξ_E → ξ').
**Phase-2 and Phase-3 model selections are completely independent.**
These keys take precedence over `dimension_reducer.types` / `surrogate.type`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reducer_type` | str | *(from `surrogate.type`)* | Reducer type for Phase 3. `"nn"` or `"pce"`. |
| `reducer_types` | list[str] | *(from `dimension_reducer.types`)* | Train multiple reducer types simultaneously. |
| `pce.order` | int | 3 | PCE polynomial degree for Phase-3 (only when `reducer_type = "pce"`). |
| `surrogate_type_to_use` | str | *(from `phase2.surrogate_type`)* | Which Phase-2 surrogate artifact to load for Phase-3 training. Defaults to the value of `phase2.surrogate_type`. Set explicitly when running Phase 3 with a pre-trained Phase-2 surrogate of a specific type (e.g. after running `scripts/train_phase2.py` independently). |

> **Why separate?**  Phase-2 learns a forward map from low-dimensional ξ' to
> settlement profiles — typically a smooth, well-conditioned regression for which
> a small NN or moderate-order PCE works well.  Phase-3 learns an *inverse*-like
> mapping from high-dimensional ξ_E to ξ', which may require very different
> model capacity.  Coupling the two types via a single `surrogate.type` key hides
> this distinction and can cause Phase-3 to silently use PCE (very slow) when
> only Phase-2 was intended to be PCE.
>
> **Important:** Before this fix, Phase-3 would incorrectly try to load a Phase-2
> surrogate whose *file name* matched the *reducer* type (e.g. `surrogate_pce.pkl`
> for a PCE reducer), even when the Phase-2 LUT was trained as a NN surrogate.
> `surrogate_type_to_use` makes this explicit and prevents stale/mismatched
> surrogate artifacts from silently producing bad Phase-3 results.

---

## `validation`

Train/validation/test split fractions (must sum to ≤ 1.0).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_fraction` | float | 0.6 | Fraction of Phase-1 data used for reducer training. |
| `val_fraction` | float | 0.2 | Fraction used for validation (early stopping). |
| `test_fraction` | float | 0.2 | Fraction held out for Phase-4 test evaluation. |

---

## `stochastic_inputs`

Optional per-sample stochastic permeability scalars appended to the Phase-1 feature vector.

By default, `k_h` and `k_v` are fixed constants taken from the `material` section.
When either flag is set to `true`, the corresponding permeability is sampled independently
for each training sample and its log value is appended to `X_train.npy`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_h` | bool | `false` | If `true`, horizontal permeability is sampled per-sample. |
| `k_v` | bool | `false` | If `true`, vertical permeability is sampled per-sample. |
| `k_h_range` | [float, float] | `[1e-13, 1e-10]` | Sampling range in m² for k_h (log-uniform). |
| `k_v_range` | [float, float] | `[1e-13, 1e-10]` | Sampling range in m² for k_v (log-uniform). |

### Feature layout

| Feature columns | Content |
|-----------------|---------|
| `0 … n_terms_E-1` | Random field coefficients ξ_E (always present). |
| `n_terms_E` | `log(k_h)` — only when `k_h: true`. |
| `n_terms_E + int(k_h)` | `log(k_v)` — only when `k_v: true`. |

Total feature width: `n_terms_E + int(k_h) + int(k_v)`.

### How stochastic scalars propagate through the pipeline

- **Phase 1**: for each sample, k_h and/or k_v are drawn log-uniformly from their
  configured ranges and used in the Biot solver call.  The log values are appended
  to `X_train.npy`.
- **Phase 2 (LUT)**: the LUT grid is extended to `effective_d = d + n_stochastic_scalars`
  dimensions.  The extra dimensions are sampled log-uniformly (not Gaussian).  The
  solver is called with the per-point k_h/k_v extracted from the scalar part of ξ'.
- **Phase 3**: the reducer maps `X_train` (width = n_terms_E + n_stochastic) to ξ'
  (width = effective_d).
- **Backward compatibility**: when both flags are `false` (default), the behaviour is
  identical to the no-stochastic case.  Old LUT config.json files without stochastic
  fields default to `n_stochastic_scalars = 0`.

Example:
```yaml
stochastic_inputs:
  k_h: true
  k_v: false
  k_h_range: [1.0e-14, 1.0e-10]
```

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

1. **Blue solid line** — Ground-truth (GT) settlement profile (full `n_nodes_x`-point curve).
2. **Red dashed line** — Reduced-space prediction profile (full `n_nodes_x`-point curve).
3. **Green open circles** — GT values at **collocation node indices** (the same indices used for Phase-3 training loss).

The x-axis always represents the full physical x-grid (0 … L_x, `n_nodes_x` points), regardless of `collocation.positions`.  Collocation positions are shown as markers, never as the x-axis itself.

> **Single source of truth**: collocation indices are computed once (in Phase 3),
> saved to `models/reduced_lut/collocation_indices.npy`, and reloaded by Phase 4
> for plotting.  Training loss and plot markers always use identical index sets.

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
│   ├── collocation_indices.npy # Node indices used for Phase-3 loss & plot markers
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


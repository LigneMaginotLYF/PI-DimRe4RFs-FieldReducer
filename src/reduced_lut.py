import numpy as np
import os
import json
import pickle
import logging
import hashlib
from datetime import date
import torch

logger = logging.getLogger(__name__)


class ReducedLUT:
    """
    Manages the reduced parameter space LUT.
    Builds d-dimensional grid, precomputes responses using polynomial or KL basis
    field reconstruction, fits surrogate, saves/loads for reuse.
    """

    _TRAIN_INDICES_FILE = 'train_indices.npy'
    _VAL_INDICES_FILE = 'val_indices.npy'

    def __init__(self, config, solver, output_dir=None):
        self.config = config
        self.solver = solver
        self.output_dir = output_dir or 'models/reduced_lut'
        os.makedirs(self.output_dir, exist_ok=True)

        lut_cfg = config.get('reduced_lut') or {}
        self.n_grid_points = lut_cfg.get('n_grid_points', 2000)
        self.grid_type = lut_cfg.get('grid_type', 'random')

        mat_cfg = config.get('material') or {}
        self.E_ref = mat_cfg.get('E_ref', 10.0e6)
        self.k_ref = mat_cfg.get('permeability_ref', 1.0e-12)
        self.k_h = mat_cfg.get('permeability_h', 1.0e-12)
        self.k_v = mat_cfg.get('permeability_v', 1.0e-12)

        sol_cfg = config.get('solver') or {}
        self.n_x = sol_cfg.get('n_nodes_x', 20)
        self.n_z = sol_cfg.get('n_nodes_z', 20)

        red_cfg = config.get('dimension_reducer') or {}
        self.d = red_cfg.get('d', 1)
        self.basis_type = red_cfg.get('basis_type', 'polynomial')
        self.basis_order = red_cfg.get('basis_order', 1)

        surr_cfg = config.get('surrogate') or {}
        self.output_representation = surr_cfg.get('output_representation', 'direct')
        self.n_output_modes = surr_cfg.get('n_output_modes', 8)
        self.bspline_degree = surr_cfg.get('bspline_degree', 3)

        stoch_cfg = config.get('stochastic_inputs') or {}
        self.k_h_stochastic = bool(stoch_cfg.get('k_h', False))
        self.k_v_stochastic = bool(stoch_cfg.get('k_v', False))
        self.k_h_range = stoch_cfg.get('k_h_range', [1e-13, 1e-10])
        self.k_v_range = stoch_cfg.get('k_v_range', [1e-13, 1e-10])
        self.n_stochastic_scalars = int(self.k_h_stochastic) + int(self.k_v_stochastic)
        # Effective LUT dimension: d (for E coefficients) + stochastic scalars
        self.effective_d = self.d + self.n_stochastic_scalars

        # Pre-compute log-space bounds used for normalization/denormalization
        self._log_k_h_lo = float(np.log(self.k_h_range[0]))
        self._log_k_h_hi = float(np.log(self.k_h_range[1]))
        self._log_k_v_lo = float(np.log(self.k_v_range[0]))
        self._log_k_v_hi = float(np.log(self.k_v_range[1]))

        self.config_hash = self._compute_config_hash()

        self.grid_points = None
        self.responses = None
        self.surrogate = None
        self.train_indices = None
        self.val_indices = None

    # ------------------------------------------------------------------
    # Normalization helpers for stochastic permeability scalar features
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_log_k(raw_log_k, lo, hi):
        """Map raw log(k) in [lo, hi] to standardized value in [-1, 1].

        Uses the midpoint/half-range formula so the entire sampling range maps
        to [-1, 1], matching a Legendre polynomial basis for uniform inputs
        and keeping feature magnitudes O(1) for numerically stable PCE/NN.

        Args:
            raw_log_k: scalar or array
            lo: float, log(k_range[0])
            hi: float, log(k_range[1])
        Returns:
            normalized value(s) in [-1, 1]
        """
        return 2.0 * (raw_log_k - lo) / (hi - lo) - 1.0

    @staticmethod
    def _denormalize_log_k(norm_v, lo, hi):
        """Map standardized value in [-1, 1] back to raw log(k) in [lo, hi].

        Inverse of :meth:`_normalize_log_k`.

        Args:
            norm_v: scalar or array in [-1, 1]
            lo: float, log(k_range[0])
            hi: float, log(k_range[1])
        Returns:
            raw log(k) value(s) in [lo, hi]
        """
        return lo + (norm_v + 1.0) / 2.0 * (hi - lo)

    def _compute_config_hash(self):
        """Compute a short hash of key parameters that affect the surrogate interface.

        The hash covers:
        - reduced space: d, basis_type, basis_order
        - solver: type (1d/2d), response_mode, n_nodes_x, n_nodes_z
        - KL reference params (nu_ref, ls_ref) when basis_type='kl'
        - surrogate output representation and n_output_modes

        Changes to any of these require recomputing the LUT.
        Physical constants (E_ref, permeability values) are intentionally excluded
        because they affect only the magnitude of the surrogate output, not its
        dimensionality or basis structure.
        """
        sol_cfg = self.config.get('solver') or {}
        rf_cfg = self.config.get('random_field') or {}
        parts = [
            f"d={self.d}",
            f"basis_type={self.basis_type}",
            f"basis_order={self.basis_order}",
            f"solver_type={sol_cfg.get('type', '2d')}",
            f"response_mode={sol_cfg.get('response_mode', 'steady_state')}",
            f"n_nodes_x={self.n_x}",
            f"n_nodes_z={self.n_z}",
            f"output_representation={self.output_representation}",
            f"n_output_modes={self.n_output_modes}",
            f"k_h_stochastic={self.k_h_stochastic}",
            f"k_v_stochastic={self.k_v_stochastic}",
        ]
        # Include parameters that affect surrogate interface/behavior but were
        # previously omitted from the hash. This ensures that cached surrogates
        # trained with different spline settings or permeability ranges do not
        # collide.
        if getattr(self, "output_representation", None) == "bspline":
            bspline_degree = getattr(self, "bspline_degree", None)
            parts.append(f"bspline_degree={bspline_degree}")
        if getattr(self, "k_h_stochastic", False):
            k_h_range = getattr(self, "k_h_range", None)
            parts.append(f"k_h_range={k_h_range}")
        if getattr(self, "k_v_stochastic", False):
            k_v_range = getattr(self, "k_v_range", None)
            parts.append(f"k_v_range={k_v_range}")
        if self.basis_type == 'kl':
            parts += [
                f"nu_ref={rf_cfg.get('nu_ref', 1.5)}",
                f"ls_ref={rf_cfg.get('length_scale_ref', 0.3)}",
            ]
        if self.basis_type == 'dct':
            # DCT basis is fixed; hash includes domain dimensions (grid already hashed
            # via n_nodes_x, n_nodes_z) and logE_std if set.
            logE_std = rf_cfg.get('logE_std', rf_cfg.get('field_fluctuation_scale', 1.0))
            parts.append(f"logE_std={logE_std}")
        if self.output_representation == 'bspline':
            # Spline degree changes the surrogate interface (number and meaning of outputs)
            parts.append(f"bspline_degree={self.bspline_degree}")
        if self.k_h_stochastic:
            # Permeability ranges change the grid sampling and normalization offset
            parts.append(f"k_h_range={self.k_h_range}")
        if self.k_v_stochastic:
            parts.append(f"k_v_range={self.k_v_range}")
        key = "_".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def generate_grid(self, seed=None):
        """Generate grid points in effective_d-dimensional reduced parameter space.

        For the first ``d`` dimensions (E-field coefficients), points are drawn
        from a standard normal distribution.  For any stochastic scalar dimensions
        (log_k_h, log_k_v), points are drawn uniformly from [-1, 1] — the
        standardized coordinate that maps the full log-space range to [-1, 1].
        This keeps feature magnitudes O(1), which is numerically beneficial for
        both PCE (Legendre-type basis) and NN models.
        """
        rng = np.random.default_rng(seed)
        if self.grid_type == 'random':
            # E-field coefficient dimensions: standard normal
            xi_E = rng.standard_normal((self.n_grid_points, self.d))
            parts = [xi_E]
            # Stochastic scalar dimensions: uniform in [-1, 1] (standardized log-space)
            if self.k_h_stochastic:
                parts.append(rng.uniform(-1, 1, size=(self.n_grid_points, 1)))
            if self.k_v_stochastic:
                parts.append(rng.uniform(-1, 1, size=(self.n_grid_points, 1)))
            self.grid_points = np.concatenate(parts, axis=1)
        else:
            n_side = int(round(self.n_grid_points ** (1 / self.effective_d))) + 1
            # Scalar dims use [-1, 1] (standardized), E-dims use [-2, 2]
            scalar_vals = np.linspace(-1, 1, n_side)
            if self.n_stochastic_scalars > 0:
                all_grids = np.meshgrid(
                    *([np.linspace(-2, 2, n_side)] * self.d
                      + [scalar_vals] * self.n_stochastic_scalars),
                    indexing='ij',
                )
                grid = np.column_stack([g.ravel() for g in all_grids])
            else:
                vals_E = np.linspace(-2, 2, n_side)
                grids_E = np.meshgrid(*([vals_E] * self.d), indexing='ij')
                grid = np.column_stack([g.ravel() for g in grids_E])
            idx = rng.choice(len(grid), size=min(self.n_grid_points, len(grid)), replace=False)
            self.grid_points = grid[idx]
        logger.info(
            f"Generated {len(self.grid_points)} grid points in "
            f"{self.effective_d}-D space (d={self.d} E-dims + "
            f"{self.n_stochastic_scalars} stochastic scalars, scalar dims in [-1,1])"
        )
        return self.grid_points

    def _reconstruct_field_polynomial(self, xi_prime):
        """
        Reconstruct material field using polynomial basis.

        Args:
            xi_prime: reduced coefficients, shape (d,)
        Returns:
            E_field: shape (n_z, n_x)
        """
        dom = self.config.get('domain', {})
        length_x = dom.get('length_x', 1.0)
        length_z = dom.get('length_z', 1.0)

        x = np.linspace(0, length_x, self.n_x)
        z = np.linspace(0, length_z, self.n_z)
        X, Z = np.meshgrid(x, z)

        x_norm = X / length_x
        z_norm = Z / length_z

        # Build the full 2D monomial basis for all x^i * z^j with i+j <= basis_order.
        # Monomials are ordered by total degree first, then by increasing z-power within
        # each degree (preserving the ordering used for basis_order <= 3).
        # Basis size = (basis_order + 1) * (basis_order + 2) // 2.
        basis_funcs = []
        for total_deg in range(0, self.basis_order + 1):
            for j in range(0, total_deg + 1):
                i = total_deg - j
                basis_funcs.append((x_norm ** i) * (z_norm ** j))

        # Validate that d doesn't exceed available basis functions for this order
        n_available = len(basis_funcs)
        if len(xi_prime) > n_available:
            raise ValueError(
                f"d={len(xi_prime)} exceeds the {n_available} basis functions available for "
                f"basis_order={self.basis_order}. "
                f"For order p the basis size is (p+1)(p+2)/2 "
                f"(e.g. order 4 → 15 terms). Reduce d or increase basis_order."
            )

        basis_funcs = basis_funcs[:len(xi_prime)]

        log_E = np.zeros_like(X)
        for i, phi in enumerate(basis_funcs):
            log_E += xi_prime[i] * phi

        return self.E_ref * np.exp(np.clip(log_E, -10, 10))

    def _reconstruct_field_kl(self, xi_prime):
        """
        Reconstruct material field using truncated KL expansion.

        The KL basis is evaluated with fixed reference Matérn parameters
        (``random_field.nu_ref`` and ``random_field.length_scale_ref`` from
        config, defaulting to 1.5 and 0.3 respectively).

        For identity-mode verification, use the same reference values in Phase 1
        by setting ``random_field.nu_sampling: false`` and
        ``random_field.length_scale_sampling: false``.

        Args:
            xi_prime: KL coefficients, shape (d,)
        Returns:
            E_field: shape (n_z, n_x)
        """
        from src.field_generator import KLExpansionField
        dom = self.config.get('domain', {})
        rf_cfg = self.config.get('random_field', {})
        length_x = dom.get('length_x', 1.0)
        length_z = dom.get('length_z', 1.0)

        nu_ref = rf_cfg.get('nu_ref', 1.5)
        ls_ref = rf_cfg.get('length_scale_ref', 0.3)

        fg = KLExpansionField(self.n_x, self.n_z, length_x, length_z)
        return fg.generate_field(xi_prime, nu_ref, ls_ref, n_terms=self.d, E_ref=self.E_ref)

    def _reconstruct_field_dct(self, xi_prime):
        """
        Reconstruct material field using the first d DCT-II basis modes.

        Uses the same fixed 2D DCT-II basis as DCTField in field_generator.py,
        ensuring consistency between Phase-1 field generation and Phase-2 LUT
        reconstruction when random_field.field_basis = "dct".

        Args:
            xi_prime: DCT coefficients, shape (d,)
        Returns:
            E_field: shape (n_z, n_x)
        """
        from src.field_generator import DCTField
        dom = self.config.get('domain', {})
        rf_cfg = self.config.get('random_field', {})
        length_x = dom.get('length_x', 1.0)
        length_z = dom.get('length_z', 1.0)
        logE_std = rf_cfg.get('logE_std', rf_cfg.get('field_fluctuation_scale', 1.0))

        fg = DCTField(self.n_x, self.n_z, length_x, length_z)
        return fg.reconstruct_from_coefficients(xi_prime, E_ref=self.E_ref, logE_std=logE_std)

    def _to_dct_space(self, Y):
        """Transform settlement profiles to DCT coefficient space.

        Args:
            Y: shape (n, n_x) - full settlement profiles
        Returns:
            B: shape (n, n_output_modes) - first n_output_modes DCT-II coefficients
        """
        from scipy.fft import dct
        B = dct(Y, type=2, norm='ortho', axis=1)
        return B[:, :self.n_output_modes]

    def _from_dct_space(self, b):
        """Reconstruct settlement profiles from DCT coefficients.

        Args:
            b: shape (n, n_output_modes) - DCT-II coefficients
        Returns:
            Y: shape (n, n_x) - reconstructed full settlement profiles
        """
        from scipy.fft import idct
        n = b.shape[0]
        B_padded = np.zeros((n, self.n_x))
        modes = min(self.n_output_modes, self.n_x)
        B_padded[:, :modes] = b[:, :modes]
        return idct(B_padded, type=2, norm='ortho', axis=1)

    def _to_poly_space(self, Y):
        """Transform settlement profiles to polynomial coefficient space.

        Fits a polynomial of degree (n_output_modes - 1) to each row of Y
        using a precomputed Vandermonde design matrix for efficiency.

        Args:
            Y: shape (n, n_x) - full settlement profiles
        Returns:
            coeffs: shape (n, n_output_modes) - polynomial coefficients in
                    descending power order (same as np.polyfit output)
        """
        n, n_x = Y.shape
        deg = self.n_output_modes - 1
        if self.n_output_modes > n_x:
            raise ValueError(
                f"Polynomial fit is ill-posed: need at least n_output_modes={self.n_output_modes} "
                f"data points but only n_x={n_x} are available. "
                "Reduce n_output_modes or increase the number of spatial nodes."
            )
        x_norm = np.linspace(0, 1, n_x)
        # Build Vandermonde matrix: shape (n_x, n_output_modes) in descending power order
        # np.polyfit convention: highest power first
        powers = np.arange(deg, -1, -1, dtype=float)  # [deg, deg-1, ..., 0]
        V = x_norm[:, None] ** powers[None, :]        # (n_x, n_output_modes)
        # Solve least-squares: V @ coeffs.T ≈ Y.T  →  coeffs = (V^+ @ Y.T).T
        coeffs, _, _, _ = np.linalg.lstsq(V, Y.T, rcond=None)
        return coeffs.T  # (n, n_output_modes)

    def _from_poly_space(self, coeffs):
        """Reconstruct settlement profiles from polynomial coefficients.

        Args:
            coeffs: shape (n, n_output_modes) - polynomial coefficients in
                    descending power order (np.polyfit convention)
        Returns:
            Y: shape (n, n_x) - reconstructed settlement profiles
        """
        x_norm = np.linspace(0, 1, self.n_x)
        deg = self.n_output_modes - 1
        powers = np.arange(deg, -1, -1, dtype=float)  # [deg, deg-1, ..., 0]
        V = x_norm[:, None] ** powers[None, :]         # (n_x, n_output_modes)
        return (V @ coeffs.T).T                        # (n, n_x)

    def _build_bspline_knots(self):
        """Build the full B-spline knot sequence for the configured settings.

        Returns:
            t_full: 1-D array of knots (including endpoint repetitions)
        """
        k = self.bspline_degree
        n_modes = self.n_output_modes
        if n_modes <= k:
            raise ValueError(
                f"n_output_modes={n_modes} must be greater than "
                f"bspline_degree={k} for B-spline representation."
            )
        n_internal = n_modes - k - 1
        if n_internal > 0:
            t_internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        else:
            t_internal = np.array([])
        t_full = np.concatenate([np.zeros(k + 1), t_internal, np.ones(k + 1)])
        return t_full

    def _to_bspline_space(self, Y):
        """Transform settlement profiles to B-spline coefficient space.

        Fits a B-spline with ``n_output_modes`` basis functions (degree
        ``bspline_degree``) to each row of Y.  The B-spline design matrix is
        precomputed once from the fixed knot sequence and x-grid, then all
        profiles are fitted via a single batched least-squares call.

        Args:
            Y: shape (n, n_x) - full settlement profiles
        Returns:
            coeffs: shape (n, n_output_modes) - B-spline coefficients
        """
        from scipy.interpolate import BSpline
        n, n_x = Y.shape
        k = self.bspline_degree
        n_modes = self.n_output_modes
        min_required = k + 1 + max(0, n_modes - k - 1)
        if n_x < min_required:
            raise ValueError(
                f"n_x={n_x} is too small for bspline with "
                f"n_output_modes={n_modes}, bspline_degree={k}. "
                f"Need at least {min_required} data points."
            )
        x_norm = np.linspace(0, 1, n_x)
        t_full = self._build_bspline_knots()
        # Build the B-spline collocation/design matrix B: (n_x, n_output_modes)
        # Each column is one B-spline basis function evaluated at all x-nodes.
        B = np.column_stack([
            BSpline.basis_element(
                t_full[i:i + k + 2], extrapolate=False
            )(x_norm)
            for i in range(n_modes)
        ])
        # Replace NaN (extrapolation outside support) with 0
        B = np.nan_to_num(B, nan=0.0)
        # Solve B @ coeffs.T ≈ Y.T  for all samples at once
        coeffs_T, _, _, _ = np.linalg.lstsq(B, Y.T, rcond=None)
        return coeffs_T.T  # (n, n_output_modes)

    def _from_bspline_space(self, coeffs):
        """Reconstruct settlement profiles from B-spline coefficients.

        Args:
            coeffs: shape (n, n_output_modes) - B-spline coefficients
        Returns:
            Y: shape (n, n_x) - reconstructed settlement profiles
        """
        from scipy.interpolate import BSpline
        n = coeffs.shape[0]
        k = self.bspline_degree
        x_norm = np.linspace(0, 1, self.n_x)
        t_full = self._build_bspline_knots()
        Y = np.zeros((n, self.n_x))
        for i in range(n):
            spl = BSpline(t_full, coeffs[i], k)
            Y[i] = spl(x_norm)
        return Y

    def _reconstruct_field(self, xi_prime):
        """
        Reconstruct material E field from d-dimensional reduced coefficients.

        When stochastic scalars are present, xi_prime may have shape
        (effective_d,) = (d + n_stochastic_scalars,).  Only the first ``d``
        components are used for E-field reconstruction; the remaining components
        encode log permeability scalars and are handled separately.

        Args:
            xi_prime: shape (effective_d,) or (d,)
        Returns:
            E_field: shape (n_z, n_x)
        """
        xi_E = xi_prime[:self.d]
        if self.basis_type == 'polynomial':
            return self._reconstruct_field_polynomial(xi_E)
        elif self.basis_type == 'kl':
            return self._reconstruct_field_kl(xi_E)
        elif self.basis_type == 'dct':
            return self._reconstruct_field_dct(xi_E)
        else:
            raise ValueError(
                f"Unknown basis_type: {self.basis_type!r}. "
                "Set dimension_reducer.basis_type in config.yaml to one of: "
                "'polynomial', 'kl', or 'dct'."
            )

    def precompute_responses(self, field_generator=None):
        """
        Run Biot solver for each grid point using basis-function field reconstruction.

        k_h and k_v are taken from config (fixed material permeabilities) unless
        the corresponding stochastic flag is set, in which case they are extracted
        from the scalar part of xi_prime (columns d and d+1 of grid_points).
        Scalar permeability dimensions are stored as standardized values in [-1, 1];
        they are denormalized back to raw log(k) and then exponentiated here.
        """
        if self.grid_points is None:
            raise RuntimeError("Call generate_grid() first")
        n = len(self.grid_points)
        self.responses = np.zeros((n, self.n_x))
        for j, xi_prime in enumerate(self.grid_points):
            if j % 100 == 0:
                logger.info(f"LUT precompute: {j}/{n}")
            E_field = self._reconstruct_field(xi_prime)
            # Extract per-point permeabilities from stochastic scalar dimensions.
            # Scalar dims are stored as standardized [-1,1] values; denormalize first.
            if self.k_h_stochastic:
                norm_v_h = float(xi_prime[self.d])
                raw_log_h = self._denormalize_log_k(norm_v_h, self._log_k_h_lo, self._log_k_h_hi)
                k_h = float(np.exp(raw_log_h))
            else:
                k_h = self.k_h
            if self.k_v_stochastic:
                norm_v_v = float(xi_prime[self.d + int(self.k_h_stochastic)])
                raw_log_v = self._denormalize_log_k(norm_v_v, self._log_k_v_lo, self._log_k_v_hi)
                k_v = float(np.exp(raw_log_v))
            else:
                k_v = self.k_v
            Y = self.solver.run(E_field, k_h, k_v)
            self.responses[j] = Y
        logger.info(f"LUT precomputation complete: {n} points")
        return self.responses

    def fit_surrogate(self, surrogate_type='nn', surrogate_cfg=None, val_fraction=0.2, seed=42):
        """
        Fit surrogate S: xi' -> Y on LUT data.
        Returns validation R².
        """
        from src.mapping_learner_nn import PhysicsDrivenMappingNN
        from src.mapping_learner_pce import PolynomialChaosExpansion

        if self.grid_points is None or self.responses is None:
            raise RuntimeError("Grid and responses must be precomputed first")

        surrogate_cfg = surrogate_cfg or {}
        n = len(self.grid_points)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self.train_indices = train_idx
        self.val_indices = val_idx

        X_train = self.grid_points[train_idx]
        Y_train = self.responses[train_idx]
        X_val = self.grid_points[val_idx]
        Y_val = self.responses[val_idx]

        # Apply output transform when output_representation != 'direct'
        if self.output_representation == 'dct':
            Y_train_fit = self._to_dct_space(Y_train)
            Y_val_fit = self._to_dct_space(Y_val)
            output_dim = min(self.n_output_modes, Y_train.shape[1])
            logger.info(
                f"Surrogate output: DCT basis with {output_dim} modes "
                f"(n_x={self.n_x})"
            )
        elif self.output_representation == 'poly':
            raise NotImplementedError(
                "Surrogate training with output_representation='poly' is not "
                "currently supported for Phase-3 reducer training: the surrogate "
                "would output polynomial coefficients, but downstream "
                "fit_with_surrogate() only reconstructs node-space for 'direct' "
                "and 'dct'. Either extend fit_with_surrogate() to handle 'poly' "
                "by reconstructing node-space, or train with 'direct'/'dct'."
            )
        elif self.output_representation == 'bspline':
            raise NotImplementedError(
                "Surrogate training with output_representation='bspline' is not "
                "currently supported for Phase-3 reducer training: the surrogate "
                "would output B-spline coefficients, but downstream "
                "fit_with_surrogate() only reconstructs node-space for 'direct' "
                "and 'dct'. Either extend fit_with_surrogate() to handle "
                "'bspline' by reconstructing node-space, or train with "
                "'direct'/'dct'."
            )
        else:
            Y_train_fit = Y_train
            Y_val_fit = Y_val
            output_dim = Y_train.shape[1]

        if surrogate_type == 'nn':
            hidden_dim = surrogate_cfg.get('hidden_dim', 64)
            n_blocks = surrogate_cfg.get('n_blocks', 3)
            epochs = surrogate_cfg.get('epochs', 200)
            lr = surrogate_cfg.get('learning_rate', 1e-3)
            batch_size = surrogate_cfg.get('batch_size', 64)

            model = PhysicsDrivenMappingNN(
                input_dim=self.effective_d, output_dim=output_dim,
                hidden_dim=hidden_dim, n_blocks=n_blocks
            )
            model.fit(X_train, Y_train_fit, X_val, Y_val_fit,
                      epochs=epochs, lr=lr, batch_size=batch_size)
            self.surrogate = model

        elif surrogate_type == 'pce':
            # PCE degree: phase2.pce.order > surrogate.basis_order
            phase2_pce_cfg = (self.config.get('phase2') or {}).get('pce') or {}
            degree = phase2_pce_cfg.get('order', surrogate_cfg.get('basis_order', 3))
            model = PolynomialChaosExpansion(
                degree=degree, n_inputs=self.effective_d, n_outputs=output_dim
            )
            model.fit(X_train, Y_train_fit)
            self.surrogate = model

        # Evaluate in full spatial domain (after inverse DCT if needed)
        Y_pred_val = self.predict(X_val)
        from sklearn.metrics import r2_score
        r2 = r2_score(Y_val, Y_pred_val)
        logger.info(f"Reduced surrogate validation R² = {r2:.4f}")

        # Roughness diagnostics: mean L2 norm of first differences
        roughness_gt = float(np.mean(
            np.sqrt(np.mean(np.diff(Y_val, axis=1) ** 2, axis=1))
        ))
        roughness_pred = float(np.mean(
            np.sqrt(np.mean(np.diff(Y_pred_val, axis=1) ** 2, axis=1))
        ))
        logger.info(
            f"Roughness ||ΔY||: GT={roughness_gt:.4e}, pred={roughness_pred:.4e}"
        )
        return r2

    def save(self, surrogate_type='nn', r2_val=None):
        """Save LUT data, surrogate, and config to output_dir."""
        np.save(os.path.join(self.output_dir, 'grid_points.npy'), self.grid_points)
        np.save(os.path.join(self.output_dir, 'responses.npy'), self.responses)

        if self.train_indices is not None:
            np.save(os.path.join(self.output_dir, self._TRAIN_INDICES_FILE), self.train_indices)
        if self.val_indices is not None:
            np.save(os.path.join(self.output_dir, self._VAL_INDICES_FILE), self.val_indices)

        if surrogate_type == 'nn' and hasattr(self.surrogate, 'state_dict'):
            torch.save(self.surrogate.state_dict(),
                       os.path.join(self.output_dir, 'surrogate_nn.pt'))
            torch.save(self.surrogate,
                       os.path.join(self.output_dir, 'surrogate_nn_full.pt'))
        else:
            with open(os.path.join(self.output_dir, 'surrogate_pce.pkl'), 'wb') as f:
                pickle.dump(self.surrogate, f)

        cfg = {
            'surrogate_type': surrogate_type,
            'input_dim': self.effective_d,
            'output_dim': int(self.responses.shape[1]),
            'n_grid_points': int(len(self.grid_points)),
            'r2_validation': float(r2_val) if r2_val is not None else None,
            'created_date': str(date.today()),
            'config_hash': self.config_hash,
            'd': self.d,
            'basis_type': self.basis_type,
            'basis_order': self.basis_order,
            'output_representation': self.output_representation,
            'n_output_modes': self.n_output_modes,
            'bspline_degree': self.bspline_degree,
            'k_h_stochastic': self.k_h_stochastic,
            'k_v_stochastic': self.k_v_stochastic,
            'n_stochastic_scalars': self.n_stochastic_scalars,
            'effective_d': self.effective_d,
        }
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Saved LUT surrogate to {self.output_dir}")

    def load(self, surrogate_type='nn'):
        """Load LUT data and surrogate from output_dir, validating config hash."""
        cfg_path = os.path.join(self.output_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            saved_hash = meta.get('config_hash')
            if saved_hash is not None and saved_hash != self.config_hash:
                raise ValueError(
                    f"Config hash mismatch: cached surrogate was built with "
                    f"d={meta.get('d')}, basis_type={meta.get('basis_type')!r}, "
                    f"basis_order={meta.get('basis_order')}, but current config has "
                    f"d={self.d}, basis_type={self.basis_type!r}, "
                    f"basis_order={self.basis_order}. "
                    f"Set reduced_lut.reuse=false to recompute."
                )

        self.grid_points = np.load(os.path.join(self.output_dir, 'grid_points.npy'))
        self.responses = np.load(os.path.join(self.output_dir, 'responses.npy'))

        if surrogate_type == 'nn':
            full_model_path = os.path.join(self.output_dir, 'surrogate_nn_full.pt')
            if os.path.exists(full_model_path):
                try:
                    self.surrogate = torch.load(full_model_path, weights_only=False)
                except TypeError:
                    # Torch < 1.13 does not accept weights_only kwarg
                    self.surrogate = torch.load(full_model_path)
            else:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                from src.mapping_learner_nn import PhysicsDrivenMappingNN
                model = PhysicsDrivenMappingNN(
                    input_dim=meta['input_dim'],
                    output_dim=meta['output_dim']
                )
                try:
                    model.load_state_dict(torch.load(
                        os.path.join(self.output_dir, 'surrogate_nn.pt'),
                        weights_only=True,
                    ))
                except TypeError:
                    model.load_state_dict(torch.load(
                        os.path.join(self.output_dir, 'surrogate_nn.pt')
                    ))
                self.surrogate = model
        else:
            with open(os.path.join(self.output_dir, 'surrogate_pce.pkl'), 'rb') as f:
                self.surrogate = pickle.load(f)

        # Restore output representation settings from saved config if present
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            saved_repr = meta.get('output_representation')
            if saved_repr is not None:
                self.output_representation = saved_repr
            saved_modes = meta.get('n_output_modes')
            if saved_modes is not None:
                self.n_output_modes = int(saved_modes)
            saved_bspline_degree = meta.get('bspline_degree')
            if saved_bspline_degree is not None:
                self.bspline_degree = int(saved_bspline_degree)
            # Restore stochastic scalar settings (default to 0 for old LUTs)
            self.k_h_stochastic = bool(meta.get('k_h_stochastic', False))
            self.k_v_stochastic = bool(meta.get('k_v_stochastic', False))
            self.n_stochastic_scalars = int(meta.get('n_stochastic_scalars', 0))
            self.effective_d = int(meta.get('effective_d', self.d))

        logger.info(f"Loaded LUT surrogate from {self.output_dir}")

        train_idx_path = os.path.join(self.output_dir, self._TRAIN_INDICES_FILE)
        if os.path.exists(train_idx_path):
            self.train_indices = np.load(train_idx_path)
        val_idx_path = os.path.join(self.output_dir, self._VAL_INDICES_FILE)
        if os.path.exists(val_idx_path):
            self.val_indices = np.load(val_idx_path)

        return self

    def predict(self, xi_prime):
        """
        Predict settlement profiles for given reduced parameters.

        The surrogate output is always converted back to node-space before
        returning, regardless of ``output_representation``:

        - ``'direct'``: surrogate outputs node-space values directly; returned as-is.
        - ``'dct'``: surrogate outputs DCT-II coefficients; reconstructed via
          inverse DCT to full spatial profiles.
        - ``'poly'``: surrogate outputs polynomial coefficients (descending power
          order); reconstructed by evaluating the polynomial at the node positions.
        - ``'bspline'``: surrogate outputs B-spline coefficients; reconstructed by
          evaluating the B-spline at the node positions.

        Args:
            xi_prime: shape (n, effective_d) or (effective_d,)
        Returns:
            Y_pred: shape (n, n_x) or (n_x,) — node-space settlement profiles
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate not fitted/loaded")
        scalar = xi_prime.ndim == 1
        if scalar:
            xi_prime = xi_prime.reshape(1, -1)
        Y_raw = self.surrogate.predict(xi_prime)
        if self.output_representation == 'dct':
            Y = self._from_dct_space(Y_raw)
        elif self.output_representation == 'poly':
            Y = self._from_poly_space(Y_raw)
        elif self.output_representation == 'bspline':
            Y = self._from_bspline_space(Y_raw)
        else:
            Y = Y_raw
        return Y[0] if scalar else Y

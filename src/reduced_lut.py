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

        lut_cfg = config.get('reduced_lut', {})
        self.n_grid_points = lut_cfg.get('n_grid_points', 2000)
        self.grid_type = lut_cfg.get('grid_type', 'random')

        mat_cfg = config.get('material', {})
        self.E_ref = mat_cfg.get('E_ref', 10.0e6)
        self.k_ref = mat_cfg.get('permeability_ref', 1.0e-12)
        self.k_h = mat_cfg.get('permeability_h', 1.0e-12)
        self.k_v = mat_cfg.get('permeability_v', 1.0e-12)

        sol_cfg = config.get('solver', {})
        self.n_x = sol_cfg.get('n_nodes_x', 20)
        self.n_z = sol_cfg.get('n_nodes_z', 20)

        red_cfg = config.get('dimension_reducer', {})
        self.d = red_cfg.get('d', 1)
        self.basis_type = red_cfg.get('basis_type', 'polynomial')
        self.basis_order = red_cfg.get('basis_order', 1)

        self.config_hash = self._compute_config_hash()

        self.grid_points = None
        self.responses = None
        self.surrogate = None
        self.train_indices = None
        self.val_indices = None

    def _compute_config_hash(self):
        """Compute a short hash of key parameters that affect the surrogate interface.

        The hash covers:
        - reduced space: d, basis_type, basis_order
        - solver: type (1d/2d), response_mode, n_nodes_x, n_nodes_z
        - KL reference params (nu_ref, ls_ref) when basis_type='kl'

        Changes to any of these require recomputing the LUT.
        Physical constants (E_ref, permeability values) are intentionally excluded
        because they affect only the magnitude of the surrogate output, not its
        dimensionality or basis structure.
        """
        sol_cfg = self.config.get('solver', {})
        rf_cfg = self.config.get('random_field', {})
        parts = [
            f"d={self.d}",
            f"basis_type={self.basis_type}",
            f"basis_order={self.basis_order}",
            f"solver_type={sol_cfg.get('type', '2d')}",
            f"response_mode={sol_cfg.get('response_mode', 'steady_state')}",
            f"n_nodes_x={self.n_x}",
            f"n_nodes_z={self.n_z}",
        ]
        if self.basis_type == 'kl':
            parts += [
                f"nu_ref={rf_cfg.get('nu_ref', 1.5)}",
                f"ls_ref={rf_cfg.get('length_scale_ref', 0.3)}",
            ]
        key = "_".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def generate_grid(self, seed=None):
        """Generate grid points in d-dimensional reduced parameter space."""
        rng = np.random.default_rng(seed)
        if self.grid_type == 'random':
            self.grid_points = rng.standard_normal((self.n_grid_points, self.d))
        else:
            n_side = int(round(self.n_grid_points ** (1 / self.d))) + 1
            vals = np.linspace(-2, 2, n_side)
            grids = np.meshgrid(*([vals] * self.d), indexing='ij')
            grid = np.column_stack([g.ravel() for g in grids])
            idx = rng.choice(len(grid), size=min(self.n_grid_points, len(grid)), replace=False)
            self.grid_points = grid[idx]
        logger.info(f"Generated {len(self.grid_points)} grid points in {self.d}-D space")
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

        basis_funcs = [np.ones_like(X)]
        if self.basis_order >= 1:
            basis_funcs.append(x_norm)
            basis_funcs.append(z_norm)
        if self.basis_order >= 2:
            basis_funcs.append(x_norm ** 2)
            basis_funcs.append(x_norm * z_norm)
            basis_funcs.append(z_norm ** 2)
        if self.basis_order >= 3:
            basis_funcs.append(x_norm ** 3)
            basis_funcs.append(x_norm ** 2 * z_norm)
            basis_funcs.append(x_norm * z_norm ** 2)
            basis_funcs.append(z_norm ** 3)

        # Validate that d doesn't exceed available basis functions for this order
        n_available = len(basis_funcs)
        if len(xi_prime) > n_available:
            raise ValueError(
                f"d={len(xi_prime)} exceeds the {n_available} basis functions available for "
                f"basis_order={self.basis_order}. Reduce d or increase basis_order."
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

    def _reconstruct_field(self, xi_prime):
        """
        Reconstruct material E field from d-dimensional reduced coefficients.

        Args:
            xi_prime: shape (d,)
        Returns:
            E_field: shape (n_z, n_x)
        """
        if self.basis_type == 'polynomial':
            return self._reconstruct_field_polynomial(xi_prime)
        elif self.basis_type == 'kl':
            return self._reconstruct_field_kl(xi_prime)
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type!r}. Use 'polynomial' or 'kl'.")

    def precompute_responses(self, field_generator=None):
        """
        Run Biot solver for each grid point using basis-function field reconstruction.
        k_h and k_v are taken from config (fixed material permeabilities).
        """
        if self.grid_points is None:
            raise RuntimeError("Call generate_grid() first")
        n = len(self.grid_points)
        self.responses = np.zeros((n, self.n_x))
        for j, xi_prime in enumerate(self.grid_points):
            if j % 100 == 0:
                logger.info(f"LUT precompute: {j}/{n}")
            E_field = self._reconstruct_field(xi_prime)
            Y = self.solver.run(E_field, self.k_h, self.k_v)
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

        output_dim = Y_train.shape[1]

        if surrogate_type == 'nn':
            hidden_dim = surrogate_cfg.get('hidden_dim', 64)
            n_blocks = surrogate_cfg.get('n_blocks', 3)
            epochs = surrogate_cfg.get('epochs', 200)
            lr = surrogate_cfg.get('learning_rate', 1e-3)
            batch_size = surrogate_cfg.get('batch_size', 64)

            model = PhysicsDrivenMappingNN(
                input_dim=self.d, output_dim=output_dim,
                hidden_dim=hidden_dim, n_blocks=n_blocks
            )
            model.fit(X_train, Y_train, X_val, Y_val,
                      epochs=epochs, lr=lr, batch_size=batch_size)
            self.surrogate = model

        elif surrogate_type == 'pce':
            degree = surrogate_cfg.get('basis_order', 3)
            model = PolynomialChaosExpansion(
                degree=degree, n_inputs=self.d, n_outputs=output_dim
            )
            model.fit(X_train, Y_train)
            self.surrogate = model

        Y_pred = self.surrogate.predict(X_val)
        from sklearn.metrics import r2_score
        r2 = r2_score(Y_val, Y_pred)
        logger.info(f"Reduced surrogate validation R² = {r2:.4f}")
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
            'input_dim': self.d,
            'output_dim': int(self.responses.shape[1]),
            'n_grid_points': int(len(self.grid_points)),
            'r2_validation': float(r2_val) if r2_val is not None else None,
            'created_date': str(date.today()),
            'config_hash': self.config_hash,
            'd': self.d,
            'basis_type': self.basis_type,
            'basis_order': self.basis_order,
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
                self.surrogate = torch.load(full_model_path, weights_only=False)
            else:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                from src.mapping_learner_nn import PhysicsDrivenMappingNN
                model = PhysicsDrivenMappingNN(
                    input_dim=meta['input_dim'],
                    output_dim=meta['output_dim']
                )
                model.load_state_dict(torch.load(
                    os.path.join(self.output_dir, 'surrogate_nn.pt')
                ))
                self.surrogate = model
        else:
            with open(os.path.join(self.output_dir, 'surrogate_pce.pkl'), 'rb') as f:
                self.surrogate = pickle.load(f)

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
        Args:
            xi_prime: shape (n, d) or (d,)
        Returns:
            Y_pred: shape (n, n_x) or (n_x,)
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate not fitted/loaded")
        scalar = xi_prime.ndim == 1
        if scalar:
            xi_prime = xi_prime.reshape(1, -1)
        Y = self.surrogate.predict(xi_prime)
        return Y[0] if scalar else Y

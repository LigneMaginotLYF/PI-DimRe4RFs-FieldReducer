import numpy as np
import os
import json
import pickle
import logging
from datetime import date
import torch

logger = logging.getLogger(__name__)


class ReducedLUT:
    """
    Manages the reduced parameter space LUT.
    Builds grid, precomputes responses, fits surrogate, saves/loads for reuse.
    """

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

        sol_cfg = config.get('solver', {})
        self.n_x = sol_cfg.get('n_nodes_x', 20)

        self.grid_points = None
        self.responses = None
        self.surrogate = None

    def generate_grid(self, seed=None):
        """Generate grid points in reduced parameter space."""
        rng = np.random.default_rng(seed)
        if self.grid_type == 'random':
            self.grid_points = rng.standard_normal((self.n_grid_points, 3))
        else:
            n_side = int(round(self.n_grid_points ** (1 / 3))) + 1
            vals = np.linspace(-2, 2, n_side)
            grid = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)
            idx = rng.choice(len(grid), size=min(self.n_grid_points, len(grid)), replace=False)
            self.grid_points = grid[idx]
        logger.info(f"Generated {len(self.grid_points)} grid points")
        return self.grid_points

    def precompute_responses(self, field_generator=None):
        """
        Run Biot solver for each grid point (constant/homogeneous field).
        Reduced field: E = E_ref * exp(xi'_0), k_h = k_ref * exp(xi'_1), k_v = k_ref * exp(xi'_2)
        """
        if self.grid_points is None:
            raise RuntimeError("Call generate_grid() first")
        n = len(self.grid_points)
        self.responses = np.zeros((n, self.n_x))
        n_z = self.config.get('solver', {}).get('n_nodes_z', 20)
        for j, xi_prime in enumerate(self.grid_points):
            if j % 100 == 0:
                logger.info(f"LUT precompute: {j}/{n}")
            E_val = self.E_ref * np.exp(np.clip(xi_prime[0], -5, 5))
            k_h = self.k_ref * np.exp(np.clip(xi_prime[1], -5, 5))
            k_v = self.k_ref * np.exp(np.clip(xi_prime[2], -5, 5))
            E_field = np.full((n_z, self.n_x), E_val)
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
                input_dim=3, output_dim=output_dim,
                hidden_dim=hidden_dim, n_blocks=n_blocks
            )
            model.fit(X_train, Y_train, X_val, Y_val,
                      epochs=epochs, lr=lr, batch_size=batch_size)
            self.surrogate = model

        elif surrogate_type == 'pce':
            degree = surrogate_cfg.get('basis_order', 3)
            model = PolynomialChaosExpansion(
                degree=degree, n_inputs=3, n_outputs=output_dim
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
            'input_dim': 3,
            'output_dim': int(self.responses.shape[1]),
            'n_grid_points': int(len(self.grid_points)),
            'r2_validation': float(r2_val) if r2_val is not None else None,
            'created_date': str(date.today()),
        }
        with open(os.path.join(self.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Saved LUT surrogate to {self.output_dir}")

    def load(self, surrogate_type='nn'):
        """Load LUT data and surrogate from output_dir."""
        self.grid_points = np.load(os.path.join(self.output_dir, 'grid_points.npy'))
        self.responses = np.load(os.path.join(self.output_dir, 'responses.npy'))

        if surrogate_type == 'nn':
            full_model_path = os.path.join(self.output_dir, 'surrogate_nn_full.pt')
            if os.path.exists(full_model_path):
                self.surrogate = torch.load(full_model_path, weights_only=False)
            else:
                cfg_path = os.path.join(self.output_dir, 'config.json')
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
        return self

    def predict(self, xi_prime):
        """
        Predict settlement profiles for given reduced parameters.
        Args:
            xi_prime: shape (n, 3) or (3,)
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

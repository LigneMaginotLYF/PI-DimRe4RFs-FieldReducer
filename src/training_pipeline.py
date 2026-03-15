import numpy as np
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the complete four-phase framework.
    """

    def __init__(self, config, output_dir=None):
        self.config = config
        if output_dir is None:
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_dir = os.path.join('results', ts)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Global artifact directories (shared across runs for reuse)
        self.data_dir = 'data'
        self.models_dir = 'models'
        self.plots_dir = 'plots'
        for d in [self.data_dir, self.models_dir, self.plots_dir,
                  os.path.join(self.plots_dir, 'material_fields'),
                  os.path.join(self.plots_dir, 'settlement_comparison'),
                  os.path.join(self.plots_dir, 'sensitivity'),
                  os.path.join(self.plots_dir, 'aggregate')]:
            os.makedirs(d, exist_ok=True)

        self.timings = {}

    def _get_solver(self):
        from src.forward_solver_2d import BiotSolver2D
        return BiotSolver2D(self.config)

    def _get_field_generator(self):
        from src.field_generator import KLExpansionField
        sol = self.config.get('solver', {})
        dom = self.config.get('domain', {})
        return KLExpansionField(
            n_nodes_x=sol.get('n_nodes_x', 20),
            n_nodes_z=sol.get('n_nodes_z', 20),
            length_x=dom.get('length_x', 1.0),
            length_z=dom.get('length_z', 1.0),
        )

    def phase1_generate_dataset(self, n_samples=None, seed=None):
        """
        Generate n_samples training samples with variable Matern KL fields.
        Returns X_train (n_samples, n_kl_terms), Y_train (n_samples, n_x).
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 1: Generating original dataset")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset', {})
        mat_cfg = self.config.get('material', {})
        rf_cfg = self.config.get('random_field', {})
        sol_cfg = self.config.get('solver', {})

        n_samples = n_samples or ds_cfg.get('n_samples', 500)
        seed = seed if seed is not None else ds_cfg.get('seed', 42)
        n_kl = ds_cfg.get('n_kl_terms_E', 5)
        E_ref = mat_cfg.get('E_ref', 10.0e6)
        k_h = mat_cfg.get('permeability_h', 1.0e-12)
        k_v = mat_cfg.get('permeability_v', 1.0e-12)
        n_x = sol_cfg.get('n_nodes_x', 20)

        nu_range = rf_cfg.get('nu_range', [0.5, 2.5])
        ls_range = rf_cfg.get('length_scale_range', [0.1, 0.5])

        rng = np.random.default_rng(seed)
        solver = self._get_solver()
        field_gen = self._get_field_generator()

        X_train = np.zeros((n_samples, n_kl))
        Y_train = np.zeros((n_samples, n_x))

        for i in range(n_samples):
            if i % 50 == 0:
                logger.info(f"Phase 1: sample {i}/{n_samples}")
            nu = rng.uniform(*nu_range)
            length_scale = rng.uniform(*ls_range)
            xi_E = rng.standard_normal(n_kl)
            E_field = field_gen.generate_field(xi_E, nu, length_scale, n_terms=n_kl, E_ref=E_ref)
            Y = solver.run(E_field, k_h, k_v)
            X_train[i] = xi_E
            Y_train[i] = Y

        np.save(os.path.join(self.data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.data_dir, 'Y_train.npy'), Y_train)

        self.timings['phase1'] = time.time() - t0
        logger.info(f"Phase 1 done in {self.timings['phase1']:.1f}s")
        logger.info(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        return X_train, Y_train

    def phase2_build_reduced_surrogate(self, seed=None, force_recompute=False):
        """
        Build LUT, fit surrogate S: xi'->Y', save for reuse.
        If surrogate artifacts already exist and force_recompute is False, loads them.
        Returns the fitted ReducedLUT object.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 2: Building reduced surrogate (LUT)")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset', {})
        surr_cfg = self.config.get('surrogate', {})
        val_cfg = self.config.get('validation', {})
        seed = seed if seed is not None else ds_cfg.get('seed', 42)

        surrogate_type = surr_cfg.get('type', 'nn')
        val_fraction = val_cfg.get('val_fraction', 0.2)

        lut_output_dir = os.path.join(self.models_dir, 'reduced_lut')
        solver = self._get_solver()

        from src.reduced_lut import ReducedLUT
        lut = ReducedLUT(self.config, solver, output_dir=lut_output_dir)

        # Skip/reuse logic: load existing artifacts if present
        config_path = os.path.join(lut_output_dir, 'config.json')
        if not force_recompute and os.path.exists(config_path):
            logger.info(f"Phase 2: Found existing surrogate at {lut_output_dir}, loading for reuse")
            lut.load(surrogate_type=surrogate_type)
            self.timings['phase2'] = time.time() - t0
            logger.info(f"Phase 2 loaded from cache in {self.timings['phase2']:.1f}s")
            return lut

        lut.generate_grid(seed=seed)
        lut.precompute_responses()
        r2_val = lut.fit_surrogate(
            surrogate_type=surrogate_type,
            surrogate_cfg=surr_cfg,
            val_fraction=val_fraction,
            seed=seed,
        )
        lut.save(surrogate_type=surrogate_type, r2_val=r2_val)

        np.save(os.path.join(self.data_dir, 'lut_grid_points.npy'), lut.grid_points)
        np.save(os.path.join(self.data_dir, 'lut_responses.npy'), lut.responses)

        self.timings['phase2'] = time.time() - t0
        logger.info(f"Phase 2 done in {self.timings['phase2']:.1f}s, R²={r2_val:.4f}")
        return lut

    def phase3_train_dimension_reducer(self, X_train, Y_train, reduced_lut, seed=None):
        """
        Train dimension reducer M: xi_E(5D) -> xi'(3D) using frozen surrogate.
        Returns trained reducer.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 3: Training dimension reducer")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset', {})
        surr_cfg = self.config.get('surrogate', {})
        val_cfg = self.config.get('validation', {})
        seed = seed if seed is not None else ds_cfg.get('seed', 42)

        surrogate_type = surr_cfg.get('type', 'nn')
        train_frac = val_cfg.get('train_fraction', 0.6)
        val_frac = val_cfg.get('val_fraction', 0.2)

        n = len(X_train)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        X_tr = X_train[train_idx]
        Y_tr = Y_train[train_idx]
        X_val = X_train[val_idx]
        Y_val = Y_train[val_idx]
        X_test = X_train[test_idx]
        Y_test = Y_train[test_idx]

        np.save(os.path.join(self.data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.data_dir, 'Y_test.npy'), Y_test)

        n_kl = X_train.shape[1]
        input_dim = n_kl
        output_dim = 3  # xi' = [E', k'_h, k'_v]

        if surrogate_type == 'nn':
            from src.mapping_learner_nn import PhysicsDrivenMappingNN
            reducer = PhysicsDrivenMappingNN(
                input_dim=input_dim, output_dim=output_dim,
                hidden_dim=surr_cfg.get('hidden_dim', 64),
                n_blocks=surr_cfg.get('n_blocks', 3),
            )
            reducer.fit_with_surrogate(
                X_tr, Y_tr, X_val, Y_val,
                surrogate=reduced_lut.surrogate,
                epochs=surr_cfg.get('epochs', 200),
                lr=surr_cfg.get('learning_rate', 1e-3),
                batch_size=surr_cfg.get('batch_size', 64),
            )
            import torch
            torch.save(reducer, os.path.join(self.models_dir, 'dimension_reducer_nn.pt'))
        else:
            from src.mapping_learner_pce import PolynomialChaosExpansion
            reducer = PolynomialChaosExpansion(
                degree=surr_cfg.get('basis_order', 3),
                n_inputs=input_dim,
                n_outputs=output_dim,
            )
            reducer.fit_with_surrogate(X_tr, Y_tr, reduced_lut.surrogate)
            import pickle
            with open(os.path.join(self.models_dir, 'dimension_reducer_pce.pkl'), 'wb') as f:
                pickle.dump(reducer, f)

        self.timings['phase3'] = time.time() - t0
        logger.info(f"Phase 3 done in {self.timings['phase3']:.1f}s")
        return reducer, (X_test, Y_test)

    def phase4_evaluate(self, reducer, reduced_lut, X_test, Y_test):
        """
        Evaluate dimension reducer on test set and produce visualizations.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 4: Test evaluation and visualization")
        logger.info("=" * 60)

        from src.validation import Validation
        from src.visualization import Visualization

        run_id = os.path.basename(self.output_dir)

        if hasattr(reducer, 'predict'):
            xi_prime_pred = reducer.predict(X_test)
        else:
            xi_prime_pred = reducer(X_test)
        Y_pred = reduced_lut.predict(xi_prime_pred)

        metrics = Validation.compute_metrics(Y_pred, Y_test)
        r2_str = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "N/A"
        logger.info(f"Test metrics: R²={r2_str}, RMSE={metrics['rmse']:.4e}, relL2={metrics['rel_l2']:.4f}")

        metrics['run_id'] = run_id
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        viz = Visualization(plots_dir=self.plots_dir)
        viz.plot_settlement_comparison(Y_test, Y_pred, n_samples=5)
        viz.plot_aggregate_metrics(Y_test, Y_pred)
        viz.plot_sobol_sensitivity(reducer, input_dim=X_test.shape[1])

        self.timings['phase4'] = time.time() - t0
        logger.info(f"Phase 4 done in {self.timings['phase4']:.1f}s")
        return metrics

    def save_run_summary(self, metrics=None):
        """Save run summary to file."""
        summary_lines = [
            "=" * 60,
            "RUN SUMMARY",
            "=" * 60,
            f"Output dir: {self.output_dir}",
            "",
            "Configuration:",
        ]
        for section, vals in self.config.items():
            if isinstance(vals, dict):
                items = ', '.join(f'{k}={v}' for k, v in vals.items())
                summary_lines.append(f"  {section}: {{{items}}}")
            else:
                summary_lines.append(f"  {section}: {vals}")
        summary_lines += [
            "",
            "Phase timings:",
        ]
        for phase, t in self.timings.items():
            summary_lines.append(f"  {phase}: {t:.1f}s")
        total = sum(self.timings.values())
        summary_lines.append(f"  TOTAL: {total:.1f}s")
        if metrics:
            summary_lines += [
                "",
                "Test metrics:",
                f"  R²: {metrics.get('r2', 'N/A')}",
                f"  RMSE: {metrics.get('rmse', 'N/A')}",
                f"  Relative L²: {metrics.get('rel_l2', 'N/A')}",
            ]
        summary_lines += [
            "",
            "Artifact locations:",
            f"  Training data:    {self.data_dir}/X_train.npy, {self.data_dir}/Y_train.npy",
            f"  LUT data:         {self.data_dir}/lut_grid_points.npy, {self.data_dir}/lut_responses.npy",
            f"  Surrogate model:  {self.models_dir}/reduced_lut/",
            f"  Reducer model:    {self.models_dir}/dimension_reducer_{{nn.pt,pce.pkl}}",
            f"  Plots:            {self.plots_dir}/",
            f"  Metrics:          {self.output_dir}/metrics.json",
            f"  Run summary:      {self.output_dir}/run_summary.txt",
        ]
        with open(os.path.join(self.output_dir, 'run_summary.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Run summary saved to {self.output_dir}/run_summary.txt")

    def orchestrate(self, phases=None):
        """
        Run all phases end-to-end.
        phases: list of phase numbers to run, default [1,2,3,4]
        """
        phases = phases or [1, 2, 3, 4]
        X_train = Y_train = None
        reduced_lut = None
        reducer = X_test = Y_test = None
        metrics = None

        if 1 in phases:
            X_train, Y_train = self.phase1_generate_dataset()
        elif any(p in phases for p in [3]):
            X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'))
            Y_train = np.load(os.path.join(self.data_dir, 'Y_train.npy'))

        if 2 in phases:
            reduced_lut = self.phase2_build_reduced_surrogate()
        elif any(p in phases for p in [3, 4]):
            from src.reduced_lut import ReducedLUT
            solver = self._get_solver()
            lut_dir = os.path.join(self.models_dir, 'reduced_lut')
            surr_type = self.config.get('surrogate', {}).get('type', 'nn')
            reduced_lut = ReducedLUT(self.config, solver, output_dir=lut_dir)
            reduced_lut.load(surrogate_type=surr_type)

        if 3 in phases:
            reducer, (X_test, Y_test) = self.phase3_train_dimension_reducer(
                X_train, Y_train, reduced_lut
            )
        elif 4 in phases:
            X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
            Y_test = np.load(os.path.join(self.data_dir, 'Y_test.npy'))
            # Load saved reducer model
            surr_type = self.config.get('surrogate', {}).get('type', 'nn')
            if surr_type == 'nn':
                import torch
                reducer_path = os.path.join(self.models_dir, 'dimension_reducer_nn.pt')
                reducer = torch.load(reducer_path, weights_only=False)
            else:
                import pickle
                reducer_path = os.path.join(self.models_dir, 'dimension_reducer_pce.pkl')
                with open(reducer_path, 'rb') as f:
                    reducer = pickle.load(f)

        if 4 in phases:
            metrics = self.phase4_evaluate(reducer, reduced_lut, X_test, Y_test)

        self.save_run_summary(metrics)
        return metrics

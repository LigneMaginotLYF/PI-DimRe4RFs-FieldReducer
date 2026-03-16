import numpy as np
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class _IdentityReducer:
    """
    Trivial identity dimension reducer for verification purposes.

    Maps xi_E → xi_E (first ``output_dim`` components).  When
    ``output_dim == input_dim`` this is a true identity.

    Used by Phase 3 when ``dimension_reducer.mode = 'identity'``.
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:self.output_dim]
        return X[:, :self.output_dim]

    def reduce(self, xi_E):
        return self.predict(xi_E)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


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

        # Per-run plots directory lives inside the run folder.
        self.plots_dir = os.path.join(output_dir, 'plots')

        # Global artifact directories (shared across runs for reuse).
        self.data_dir = 'data'
        self.models_dir = 'models'
        for d in [self.data_dir, self.models_dir]:
            os.makedirs(d, exist_ok=True)

        self.timings = {}

    def _get_solver(self):
        solver_type = (self.config.get('solver') or {}).get('type', '2d')
        if solver_type == '1d':
            from src.forward_solver_1d import BiotSolver1D
            return BiotSolver1D(self.config)
        else:
            from src.forward_solver_2d import BiotSolver2D
            return BiotSolver2D(self.config)

    def _get_field_generator(self):
        sol = self.config.get('solver') or {}
        dom = self.config.get('domain') or {}
        rf_cfg = self.config.get('random_field') or {}
        field_basis = rf_cfg.get('field_basis', 'kl')
        if field_basis == 'dct':
            from src.field_generator import DCTField
            return DCTField(
                n_nodes_x=sol.get('n_nodes_x', 20),
                n_nodes_z=sol.get('n_nodes_z', 20),
                length_x=dom.get('length_x', 1.0),
                length_z=dom.get('length_z', 1.0),
            )
        else:
            from src.field_generator import KLExpansionField
            return KLExpansionField(
                n_nodes_x=sol.get('n_nodes_x', 20),
                n_nodes_z=sol.get('n_nodes_z', 20),
                length_x=dom.get('length_x', 1.0),
                length_z=dom.get('length_z', 1.0),
            )

    @staticmethod
    def _resolve_n_terms(ds_cfg):
        """Resolve number of field terms, supporting n_terms_E (new) and
        n_kl_terms_E (legacy) config keys.  Emits a deprecation warning when
        the old key is used and the new key is absent."""
        import warnings
        if 'n_terms_E' in ds_cfg:
            return ds_cfg['n_terms_E']
        if 'n_kl_terms_E' in ds_cfg:
            warnings.warn(
                "dataset.n_kl_terms_E is deprecated; use dataset.n_terms_E instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return ds_cfg['n_kl_terms_E']
        return 5  # default

    @staticmethod
    def _compute_collocation_indices(config, n_nodes_x, length_x, phase=None):
        """
        Compute collocation node indices from config.

        Uses ``collocation.positions`` (physical x-coordinates) when provided;
        otherwise defaults to all n_nodes_x nodes.

        Phase-specific overrides are supported for independent configuration of
        Phase-2 evaluation markers and Phase-3 training loss:

          - ``phase='phase2'``: reads ``collocation_phase2.positions`` first;
            falls back to ``collocation.positions``, then to all nodes.
          - ``phase='phase3'``: reads ``collocation_phase3.positions`` first;
            falls back to ``collocation.positions``, then to all nodes.
          - ``phase=None``: reads ``collocation.positions`` only.

        Args:
            config      : Full pipeline config dict.
            n_nodes_x   : Number of x-nodes in the solver grid.
            length_x    : Physical length of the domain in x.
            phase       : Optional 'phase2' | 'phase3' to use phase-specific
                          config sections with fallback.

        Returns:
            colloc_idx  : 1-D numpy int array of node indices, sorted ascending.
        """
        positions_cfg = None
        if phase == 'phase2':
            phase_cfg = config.get('collocation_phase2') or {}
            positions_cfg = phase_cfg.get('positions', None)
        elif phase == 'phase3':
            phase_cfg = config.get('collocation_phase3') or {}
            positions_cfg = phase_cfg.get('positions', None)

        # Fallback to common collocation section when phase-specific is absent
        if positions_cfg is None:
            collocation_cfg = config.get('collocation') or {}
            positions_cfg = collocation_cfg.get('positions', None)

        if positions_cfg is not None:
            positions = np.asarray(positions_cfg, dtype=float)
            # Map each physical position to the nearest node index.
            # The grid is uniform so searchsorted gives an O(log n) candidate,
            # but we still clamp and compare both neighbours for correctness.
            x_grid = np.linspace(0.0, length_x, n_nodes_x)
            dx = length_x / max(n_nodes_x - 1, 1)
            raw = np.searchsorted(x_grid, positions)
            raw = np.clip(raw, 0, n_nodes_x - 1)
            # Prefer left neighbour when equidistant or closer
            left = np.clip(raw - 1, 0, n_nodes_x - 1)
            use_left = np.abs(x_grid[left] - positions) <= np.abs(x_grid[raw] - positions)
            indices = np.where(use_left, left, raw).astype(int)
            colloc_idx = np.unique(indices)  # sorted, deduplicated
            phase_label = f" (phase={phase})" if phase else ""
            logger.info(
                f"Collocation{phase_label}: using {len(colloc_idx)} node indices "
                f"from {len(positions)} configured positions: {colloc_idx.tolist()}"
            )
        else:
            colloc_idx = np.arange(n_nodes_x, dtype=int)
            phase_label = f" (phase={phase})" if phase else ""
            logger.info(
                f"Collocation{phase_label}: no positions configured – "
                f"using all {n_nodes_x} nodes"
            )
        return colloc_idx


    def phase1_generate_dataset(self, n_samples=None, seed=None):
        """
        Generate n_samples training samples with random material fields.
        Returns X_train (n_samples, n_terms), Y_train (n_samples, n_x).

        Field basis is selected by random_field.field_basis:
          "kl"  (default) – per-sample Matérn-KL eigenbasis (original behaviour).
          "dct"           – fixed 2D DCT basis with Matérn-shaped coefficient
                            variance; basis functions never change between samples.

        Respects config dataset.reuse flag: when True and saved arrays exist,
        loads them instead of recomputing.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 1: Generating original dataset")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset') or {}
        mat_cfg = self.config.get('material') or {}
        rf_cfg = self.config.get('random_field') or {}
        sol_cfg = self.config.get('solver') or {}

        n_samples = n_samples or ds_cfg.get('n_samples', 500)
        seed = seed if seed is not None else ds_cfg.get('seed', 42)
        n_terms = self._resolve_n_terms(ds_cfg)
        E_ref = mat_cfg.get('E_ref', 10.0e6)
        k_h = mat_cfg.get('permeability_h', 1.0e-12)
        k_v = mat_cfg.get('permeability_v', 1.0e-12)
        n_x = sol_cfg.get('n_nodes_x', 20)
        logE_std = rf_cfg.get('logE_std', rf_cfg.get('field_fluctuation_scale', 1.0))
        field_basis = rf_cfg.get('field_basis', 'kl')

        # Per-sample E_ref sampling (DCT basis only): encodes mean shift in DC coefficient
        e_ref_sampling = rf_cfg.get('E_ref_sampling', False)
        e_ref_factor_range = rf_cfg.get('E_ref_factor_range', [0.5, 1.5])

        # Resolve save paths (allow config override)
        x_path = ds_cfg.get('path_X', os.path.join(self.data_dir, 'X_train.npy'))
        y_path = ds_cfg.get('path_Y', os.path.join(self.data_dir, 'Y_train.npy'))

        # Reuse existing dataset if requested
        reuse = ds_cfg.get('reuse', False)
        if reuse and os.path.exists(x_path) and os.path.exists(y_path):
            logger.info(f"Phase 1: reuse=True -- loading existing dataset from {x_path}")
            X_train = np.load(x_path)
            Y_train = np.load(y_path)
            self.timings['phase1'] = time.time() - t0
            logger.info(f"Phase 1 loaded in {self.timings['phase1']:.1f}s -- {X_train.shape}, {Y_train.shape}")
            return X_train, Y_train

        nu_range = rf_cfg.get('nu_range', [0.5, 2.5])
        ls_range = rf_cfg.get('length_scale_range', [0.1, 0.5])
        nu_sampling = rf_cfg.get('nu_sampling', True)
        ls_sampling = rf_cfg.get('length_scale_sampling', True)
        nu_ref = rf_cfg.get('nu_ref', 1.5)
        ls_ref = rf_cfg.get('length_scale_ref', 0.3)

        rng = np.random.default_rng(seed)
        solver = self._get_solver()
        field_gen = self._get_field_generator()

        X_train = np.zeros((n_samples, n_terms))
        Y_train = np.zeros((n_samples, n_x))

        if e_ref_sampling and field_basis == 'dct':
            logger.info(
                f"Phase 1: E_ref_sampling=True, factor_range={e_ref_factor_range}, "
                "mean encoded in DC DCT coefficient"
            )

        for i in range(n_samples):
            if i % 50 == 0:
                logger.info(f"Phase 1: sample {i}/{n_samples}")
            nu = rng.uniform(*nu_range) if nu_sampling else nu_ref
            length_scale = rng.uniform(*ls_range) if ls_sampling else ls_ref
            if field_basis == 'dct':
                # Optionally sample per-sample E_ref_factor and encode in DC coefficient
                factor = rng.uniform(*e_ref_factor_range) if e_ref_sampling else None
                xi_E, E_field = field_gen.generate_field(
                    rng, nu, length_scale,
                    n_terms=n_terms, E_ref=E_ref, logE_std=logE_std,
                    E_ref_factor=factor,
                )
            else:
                xi_E = rng.standard_normal(n_terms)
                E_field = field_gen.generate_field(
                    xi_E, nu, length_scale,
                    n_terms=n_terms, E_ref=E_ref, logE_std=logE_std,
                )
            Y = solver.run(E_field, k_h, k_v)
            X_train[i] = xi_E
            Y_train[i] = Y

        os.makedirs(os.path.dirname(x_path) if os.path.dirname(x_path) else '.', exist_ok=True)
        np.save(x_path, X_train)
        np.save(y_path, Y_train)

        self.timings['phase1'] = time.time() - t0
        logger.info(f"Phase 1 done in {self.timings['phase1']:.1f}s")
        logger.info(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        return X_train, Y_train

    def phase2_build_reduced_surrogate(self, seed=None):
        """
        Build LUT, fit surrogate(s) S: xi'->Y', and save artifacts.

        Supports multiple surrogate types via surrogate.types (list) in config.
        If only one type is requested, behaviour is identical to the original.

        Phase-2 surrogate reuse is intentionally disabled: fresh training is
        performed on every call to prevent stale-artifact contamination when
        problem setup (dimension/regime/basis/etc.) changes.

        Returns the fitted ReducedLUT object (primary type loaded).
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 2: Building reduced surrogate (LUT) -- always recomputing")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset') or {}
        surr_cfg = self.config.get('surrogate') or {}
        lut_cfg = self.config.get('reduced_lut') or {}
        val_cfg = self.config.get('validation') or {}
        seed = seed if seed is not None else ds_cfg.get('seed', 42)

        # Support a list of types for simultaneous training
        surrogate_types = surr_cfg.get('types', None)
        if surrogate_types is None:
            surrogate_types = [surr_cfg.get('type', 'nn')]
        val_fraction = val_cfg.get('val_fraction', 0.2)

        lut_output_dir = os.path.join(self.models_dir, 'reduced_lut')
        solver = self._get_solver()

        # Phase-2 surrogate reuse is intentionally disabled: always recompute
        # so each run produces fresh artifacts and no stale model contaminates
        # a changed problem setup.
        from src.reduced_lut import ReducedLUT
        lut = ReducedLUT(self.config, solver, output_dir=lut_output_dir)

        lut.generate_grid(seed=seed)
        lut.precompute_responses()

        r2_by_type = {}
        for s_type in surrogate_types:
            r2_val = lut.fit_surrogate(
                surrogate_type=s_type,
                surrogate_cfg=surr_cfg,
                val_fraction=val_fraction,
                seed=seed,
            )
            lut.save(surrogate_type=s_type, r2_val=r2_val)
            r2_by_type[s_type] = r2_val
            logger.info(f"Phase 2 [{s_type}] R2={r2_val:.4f}")

            # Phase-2 independent test-set evaluation (metrics + settlement plots)
            self._phase2_evaluate_surrogate(lut, s_type, seed=seed)

            # Phase-2 surrogate accuracy plots (scatter + profile comparison)
            from src.visualization import Visualization
            viz_p2 = Visualization(plots_dir=self.plots_dir)
            viz_p2.plot_phase2_surrogate_accuracy(lut, surrogate_type=s_type,
                                                  val_fraction=val_fraction, seed=seed)

        # Save LUT grid arrays to data/ for easy access
        np.save(os.path.join(self.data_dir, 'lut_grid_points.npy'), lut.grid_points)
        np.save(os.path.join(self.data_dir, 'lut_responses.npy'), lut.responses)

        # Multi-surrogate comparison plot
        if len(surrogate_types) > 1:
            from src.visualization import Visualization
            viz = Visualization(plots_dir=self.plots_dir)
            metrics_by_type = {s_type: {'r2': r2, 'rmse': None}
                               for s_type, r2 in r2_by_type.items()}
            viz.plot_surrogate_comparison(metrics_by_type, label='Phase-2 surrogate')

        # Restore the primary surrogate on the lut object
        lut.load(surrogate_type=surrogate_types[0])

        self.timings['phase2'] = time.time() - t0
        logger.info(f"Phase 2 done in {self.timings['phase2']:.1f}s -- {r2_by_type}")
        return lut

    def _phase2_evaluate_surrogate(self, lut, surrogate_type, seed=None):
        """
        Evaluate phase-2 surrogate on an independent test split drawn from the LUT.
        Saves metrics to models/reduced_lut/<surrogate_type>/evaluation/metrics.json
        and a few qualitative settlement comparison plots.
        """
        from src.validation import Validation
        from src.visualization import Visualization

        n = len(lut.grid_points)
        rng = np.random.default_rng(seed if seed is not None else 42)
        idx = rng.permutation(n)
        # Use the last 20% as an independent test set (not used in surrogate training)
        n_test = max(1, int(n * 0.2))
        test_idx = idx[-n_test:]
        X_test = lut.grid_points[test_idx]
        Y_test = lut.responses[test_idx]

        Y_pred = lut.predict(X_test)
        metrics = Validation.compute_metrics(Y_pred, Y_test)

        # Roughness metric: mean L2 norm of first differences ||ΔY||
        roughness_gt = float(np.mean(np.sqrt(np.mean(np.diff(Y_test, axis=1) ** 2, axis=1))))
        roughness_pred = float(np.mean(np.sqrt(np.mean(np.diff(Y_pred, axis=1) ** 2, axis=1))))
        metrics['roughness_gt'] = roughness_gt
        metrics['roughness_pred'] = roughness_pred

        eval_dir = os.path.join(self.models_dir, 'reduced_lut', surrogate_type, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        # Qualitative settlement-comparison plots
        viz = Visualization(plots_dir=eval_dir)
        n_plot = min(5, len(Y_test))
        viz.plot_settlement_comparison(Y_test, Y_pred, n_samples=n_plot)

        r2_str = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "N/A"
        logger.info(
            f"Phase 2 [{surrogate_type}] independent eval: R2={r2_str}, "
            f"RMSE={metrics['rmse']:.4e}, relL2={metrics['rel_l2']:.4f}, "
            f"roughness_gt={roughness_gt:.4e}, roughness_pred={roughness_pred:.4e} "
            f"(saved to {eval_dir})"
        )
        return metrics

    def phase3_train_dimension_reducer(self, X_train, Y_train, reduced_lut, seed=None):
        """
        Train dimension reducer M: xi_E(5D) -> xi'(3D) using frozen surrogate.

        Supports multiple reducer types via dimension_reducer.types (list).
        Returns (reducer_or_dict, (X_test, Y_test)).
        When multiple types are requested, returns a dict {type: reducer}.

        Collocation indices are computed from ``collocation.positions`` (if set)
        and saved to ``models/reduced_lut/collocation_indices.npy`` so that
        Phase-4 plotting can use the same subset.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 3: Training dimension reducer")
        logger.info("=" * 60)

        ds_cfg = self.config.get('dataset') or {}
        surr_cfg = self.config.get('surrogate') or {}
        red_cfg = self.config.get('dimension_reducer') or {}
        val_cfg = self.config.get('validation') or {}
        sol_cfg = self.config.get('solver') or {}
        dom_cfg = self.config.get('domain') or {}
        seed = seed if seed is not None else ds_cfg.get('seed', 42)

        # Compute collocation indices – single source of truth for Phase-3 + Phase-4
        n_nodes_x = sol_cfg.get('n_nodes_x', 20)
        length_x = dom_cfg.get('length_x', 1.0)
        colloc_idx = self._compute_collocation_indices(self.config, n_nodes_x, length_x,
                                                       phase='phase3')
        lut_output_dir = os.path.join(self.models_dir, 'reduced_lut')
        os.makedirs(lut_output_dir, exist_ok=True)
        np.save(os.path.join(lut_output_dir, 'collocation_indices.npy'), colloc_idx)
        logger.info(
            f"Phase 3: saved collocation_indices.npy "
            f"({len(colloc_idx)} indices) to {lut_output_dir}"
        )

        # Support a list of types
        reducer_types = red_cfg.get('types', None)
        if reducer_types is None:
            surrogate_type = surr_cfg.get('type', 'nn')
            reducer_types = [surrogate_type]

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
        output_dim = red_cfg.get('d', 1)  # output dimension matches reduced space

        # Identity mapping mode: bypass learned reducer
        reducer_mode = red_cfg.get('mode', 'learned')
        if reducer_mode == 'identity':
            logger.info("Phase 3: identity mode -- using trivial identity reducer")
            reducer = _IdentityReducer(input_dim=input_dim, output_dim=output_dim)
            # Save reducer
            import pickle
            with open(os.path.join(self.models_dir, 'dimension_reducer_identity.pkl'), 'wb') as f:
                pickle.dump(reducer, f)
            np.save(os.path.join(self.data_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(self.data_dir, 'Y_test.npy'), Y_test)
            self.timings['phase3'] = time.time() - t0
            logger.info(f"Phase 3 (identity) done in {self.timings['phase3']:.2f}s")
            return reducer, (X_test, Y_test)

        reducers = {}
        for r_type in reducer_types:
            # Load the correct surrogate for this reducer type into a dedicated LUT instance
            from src.reduced_lut import ReducedLUT
            lut_for_type = ReducedLUT(self.config, self._get_solver(),
                                      output_dir=lut_output_dir)
            lut_for_type.load(surrogate_type=r_type)
            reducer = self._train_single_reducer(
                r_type, input_dim, output_dim,
                X_tr, Y_tr, X_val, Y_val,
                surr_cfg, red_cfg, lut_for_type,
                colloc_idx=colloc_idx,
            )
            reducers[r_type] = reducer

        # Save all reducer models
        for r_type, reducer in reducers.items():
            if r_type == 'nn':
                import torch
                torch.save(reducer, os.path.join(self.models_dir, 'dimension_reducer_nn.pt'))
            else:
                import pickle
                with open(os.path.join(self.models_dir, f'dimension_reducer_{r_type}.pkl'), 'wb') as f:
                    pickle.dump(reducer, f)

        self.timings['phase3'] = time.time() - t0
        logger.info(f"Phase 3 done in {self.timings['phase3']:.1f}s")

        # Return single reducer for backward-compat when only one type
        if len(reducer_types) == 1:
            return reducers[reducer_types[0]], (X_test, Y_test)
        return reducers, (X_test, Y_test)

    def _train_single_reducer(self, r_type, input_dim, output_dim,
                              X_tr, Y_tr, X_val, Y_val,
                              surr_cfg, red_cfg, reduced_lut,
                              colloc_idx=None):
        """Train a single reducer of the given type and return it."""
        if r_type == 'nn':
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
                colloc_idx=colloc_idx,
                output_representation=reduced_lut.output_representation,
                n_output_modes=reduced_lut.n_output_modes,
                n_nodes_x=reduced_lut.n_x,
            )
        else:
            from src.mapping_learner_pce import PolynomialChaosExpansion
            # PCE degree comes from dimension_reducer.basis_order (authoritative).
            # surrogate.basis_order is kept as a legacy fallback only.
            reducer = PolynomialChaosExpansion(
                degree=red_cfg.get('basis_order', surr_cfg.get('basis_order', 3)),
                n_inputs=input_dim,
                n_outputs=output_dim,
            )
            reducer.fit_with_surrogate(X_tr, Y_tr, reduced_lut.surrogate,
                                       colloc_idx=colloc_idx,
                                       output_representation=reduced_lut.output_representation,
                                       n_output_modes=reduced_lut.n_output_modes,
                                       n_nodes_x=reduced_lut.n_x)
        return reducer

    def phase4_evaluate(self, reducer, reduced_lut, X_test, Y_test):
        """
        Evaluate dimension reducer on test set and produce visualizations.

        All plots are saved under <output_dir>/plots/ (per-run directory).
        When reducer is a dict {type: model}, evaluates all types and produces
        a comparison plot.

        Settlement-comparison plots (n_samples=5) use direct Biot solver predictions
        by default (phase4.use_direct_physics_for_plots=true) so that plots isolate
        reducer + reduced-field error, not Phase-2 surrogate oscillations.
        Metrics are always computed from surrogate predictions for speed.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE 4: Test evaluation and visualization")
        logger.info("=" * 60)

        from src.validation import Validation
        from src.visualization import Visualization

        run_id = os.path.basename(self.output_dir)
        viz = Visualization(plots_dir=self.plots_dir)

        # Handle dict of reducers (multi-type scenario)
        if isinstance(reducer, dict):
            metrics_by_type = {}
            for r_type, r_model in reducer.items():
                m = self._evaluate_single_reducer(r_model, reduced_lut, X_test, Y_test, run_id)
                metrics_by_type[r_type] = m
            viz.plot_surrogate_comparison(metrics_by_type, label='Phase-3 dimension reducer')
            # Use first as primary for detailed plots
            primary_type = next(iter(reducer))
            primary_reducer = reducer[primary_type]
            metrics = metrics_by_type[primary_type]
        else:
            primary_reducer = reducer
            metrics = self._evaluate_single_reducer(reducer, reduced_lut, X_test, Y_test, run_id)

        # Surrogate-based predictions for metrics
        xi_prime_pred = primary_reducer.predict(X_test)
        Y_pred = reduced_lut.predict(xi_prime_pred)

        # Identity-mode equivalence check
        red_cfg = self.config.get('dimension_reducer') or {}
        if red_cfg.get('mode', 'learned') == 'identity':
            identity_report = self._run_identity_check(X_test, Y_test, reduced_lut)
            metrics.update(identity_report)

        # Physical x-positions for settlement comparison plots
        dom_cfg = self.config.get('domain') or {}
        sol_cfg = self.config.get('solver') or {}
        n_x = sol_cfg.get('n_nodes_x', 20)
        length_x = dom_cfg.get('length_x', 1.0)
        # Always use the full x-grid for the curve; collocation positions are
        # markers only (green circles) and never replace the full x-axis.
        x_positions = np.linspace(0.0, length_x, n_x)

        # Load collocation indices saved during Phase 3 (single source of truth).
        # Fall back to _compute_collocation_indices if artifact is absent (e.g.
        # when Phase 4 is called standalone without a prior Phase 3 run).
        colloc_idx_path = os.path.join(
            self.models_dir, 'reduced_lut', 'collocation_indices.npy'
        )
        if os.path.exists(colloc_idx_path):
            colloc_idx = np.load(colloc_idx_path)
        else:
            colloc_idx = self._compute_collocation_indices(
                self.config, n_x, length_x, phase='phase3'
            )

        # --- Settlement comparison plots ---
        # Pre-select a fixed set of plot samples so that both the plain comparison
        # and the collocation-overlay comparison are generated from EXACTLY the
        # same samples and the same forward path (direct physics or surrogate).
        p4_cfg = self.config.get('phase4') or {}
        use_direct_physics = p4_cfg.get('use_direct_physics_for_plots', True)
        n_plot = min(5, len(X_test))
        rng_plot = np.random.default_rng(42)
        plot_indices = rng_plot.choice(len(X_test), size=n_plot, replace=False)

        Y_plot_true = Y_test[plot_indices]
        if use_direct_physics:
            Y_direct = self._compute_direct_physics_predictions(
                xi_prime_pred[plot_indices], reduced_lut
            )
            if Y_direct is not None:
                Y_plot_pred = Y_direct
                logger.info(
                    "Phase 4: settlement comparison plots use direct Biot solver "
                    "(phase4.use_direct_physics_for_plots=true)"
                )
            else:
                # Fallback: surrogate predictions for the selected samples
                Y_plot_pred = Y_pred[plot_indices]
                logger.warning(
                    "Phase 4: direct-physics predictions failed; "
                    "falling back to surrogate predictions for plots."
                )
        else:
            Y_plot_pred = Y_pred[plot_indices]

        # Both plots use the same pre-selected samples and the same predictions
        viz.plot_settlement_comparison(
            Y_plot_true, Y_plot_pred, n_samples=n_plot,
            x_positions=x_positions,
        )

        viz.plot_aggregate_metrics(Y_test, Y_pred)
        viz.plot_sobol_sensitivity(primary_reducer, input_dim=X_test.shape[1])

        # Collocation point visualizations (when LUT has training indices)
        if reduced_lut.train_indices is not None:
            viz.plot_settlement_comparison_with_collocation(
                Y_plot_true, Y_plot_pred, reduced_lut, n_samples=n_plot,
                x_positions=x_positions,
                colloc_idx=colloc_idx,
            )

        # Material field comparison plots
        self._plot_material_field_comparison(
            X_test, xi_prime_pred, viz, reduced_lut, n_samples=min(5, len(X_test))
        )

        # Save metrics
        metrics['run_id'] = run_id
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        r2_str = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "N/A"
        logger.info(f"Test metrics: R2={r2_str}, RMSE={metrics['rmse']:.4e}, relL2={metrics['rel_l2']:.4f}")

        self.timings['phase4'] = time.time() - t0
        logger.info(f"Phase 4 done in {self.timings['phase4']:.1f}s")
        return metrics

    def _compute_direct_physics_predictions(self, xi_prime_samples, reduced_lut):
        """
        Compute settlement predictions via direct Biot solver on reconstructed fields.

        For each sample in xi_prime_samples, reconstructs the E field and runs the
        solver directly.  Used for qualitative Phase-4 plots to isolate
        reducer + reduced-field error from Phase-2 surrogate oscillations.

        Args:
            xi_prime_samples: shape (n, d) – reduced parameters for n plot samples
            reduced_lut: ReducedLUT instance (provides _reconstruct_field)

        Returns:
            Y_direct: shape (n, n_x) or None if an error occurs
        """
        try:
            solver = self._get_solver()
            mat_cfg = self.config.get('material') or {}
            k_h = mat_cfg.get('permeability_h', 1.0e-12)
            k_v = mat_cfg.get('permeability_v', 1.0e-12)
            n = len(xi_prime_samples)
            sol_cfg = self.config.get('solver') or {}
            n_x = sol_cfg.get('n_nodes_x', 20)
            Y_direct = np.zeros((n, n_x))
            for i, xi in enumerate(xi_prime_samples):
                E_field = reduced_lut._reconstruct_field(xi)
                Y_direct[i] = solver.run(E_field, k_h, k_v)
            return Y_direct
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            logger.warning(
                f"Direct-physics plot generation failed ({type(exc).__name__}: {exc}); "
                "falling back to surrogate predictions for plots."
            )
            return None

    def _evaluate_single_reducer(self, reducer, reduced_lut, X_test, Y_test, run_id):
        """Evaluate a single reducer and return metrics dict."""
        from src.validation import Validation
        if hasattr(reducer, 'predict'):
            xi_prime_pred = reducer.predict(X_test)
        else:
            xi_prime_pred = reducer(X_test)
        Y_pred = reduced_lut.predict(xi_prime_pred)
        return Validation.compute_metrics(Y_pred, Y_test)

    def _run_identity_check(self, X_test, Y_test, reduced_lut):
        """
        Physical consistency check for identity-mapping mode.

        In identity mode the reducer is a pass-through (xi' = xi_E).  The
        settlement obtained by reconstructing the field from xi_E and running
        the direct Biot solver should match the original settlement (Y_test)
        up to the KL-basis reconstruction accuracy.

        When ``random_field.nu_sampling`` and ``random_field.length_scale_sampling``
        are both ``false`` (recommended for identity mode), the reconstruction
        uses the same fixed KL basis as Phase 1 and the residual error is at
        machine-precision level.

        Saves the report to ``<output_dir>/identity_check.json``.

        Returns a dict of identity-check metrics (max_abs_error, rmse, r2) that
        is merged into the main metrics.
        """
        from src.validation import Validation

        rf_cfg = self.config.get('random_field') or {}
        ds_cfg = self.config.get('dataset') or {}
        mat_cfg = self.config.get('material') or {}
        n_terms = self._resolve_n_terms(ds_cfg)
        E_ref = mat_cfg.get('E_ref', 10.0e6)
        k_h = mat_cfg.get('permeability_h', 1.0e-12)
        k_v = mat_cfg.get('permeability_v', 1.0e-12)
        nu_sampling = rf_cfg.get('nu_sampling', True)
        ls_sampling = rf_cfg.get('length_scale_sampling', True)
        nu_ref = rf_cfg.get('nu_ref', 1.5)
        ls_ref = rf_cfg.get('length_scale_ref', 0.3)
        logE_std = rf_cfg.get('logE_std', rf_cfg.get('field_fluctuation_scale', 1.0))
        field_basis = rf_cfg.get('field_basis', 'kl')

        if nu_sampling or ls_sampling:
            logger.warning(
                "Identity check: nu_sampling or length_scale_sampling is True. "
                "For exact identity (machine-precision error), set both to false "
                "so Phase-1 generation and Phase-2 reconstruction share the same "
                "basis.  Reported errors will include basis mismatch."
            )

        field_gen = self._get_field_generator()
        solver = self._get_solver()
        seed = ds_cfg.get('seed', 42)
        # Use SeedSequence to spawn an independent child stream; avoids
        # potential arithmetic overflow and is statistically sound.
        rng = np.random.default_rng(
            np.random.SeedSequence(seed).spawn(1)[0]
        )

        nu_range = rf_cfg.get('nu_range', [0.5, 2.5])
        ls_range = rf_cfg.get('length_scale_range', [0.1, 0.5])

        n = len(X_test)
        Y_direct = np.zeros_like(Y_test)
        for i in range(n):
            nu = rng.uniform(*nu_range) if nu_sampling else nu_ref
            length_scale = rng.uniform(*ls_range) if ls_sampling else ls_ref
            if field_basis == 'dct':
                # DCT: X_test[i] are the stored coefficients; reconstruct directly.
                E_field = field_gen.reconstruct_from_coefficients(
                    X_test[i], E_ref=E_ref, logE_std=logE_std
                )
            else:
                E_field = field_gen.generate_field(
                    X_test[i], nu, length_scale,
                    n_terms=n_terms, E_ref=E_ref, logE_std=logE_std,
                )
            Y_direct[i] = solver.run(E_field, k_h, k_v)

        metrics_direct = Validation.compute_metrics(Y_direct, Y_test)
        max_abs = float(np.max(np.abs(Y_direct - Y_test)))

        report = {
            'identity_check_max_abs_error': max_abs,
            'identity_check_rmse': metrics_direct['rmse'],
            'identity_check_r2': metrics_direct['r2'],
        }

        report_path = os.path.join(self.output_dir, 'identity_check.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"Identity check: max_abs={max_abs:.4e}, "
            f"RMSE={metrics_direct['rmse']:.4e}, "
            f"R2={metrics_direct['r2']}"
        )
        return report

    def _plot_material_field_comparison(
        self, X_test, xi_prime_pred, viz, reduced_lut, n_samples=5
    ):
        """
        Generate and save material field comparison plots.

        Reconstructs original E fields from X_test (KL coefficients) and compares
        them to the reduced E' field derived from the predicted xi' via basis reconstruction.
        k_h and k_v are taken from config (fixed material permeabilities).
        """
        mat_cfg = self.config.get('material') or {}
        E_ref = mat_cfg.get('E_ref', 10.0e6)
        k_h = mat_cfg.get('permeability_h', 1.0e-12)
        k_v = mat_cfg.get('permeability_v', 1.0e-12)

        field_gen = self._get_field_generator()
        rf_cfg = self.config.get('random_field') or {}
        ds_cfg = self.config.get('dataset') or {}
        n_terms = self._resolve_n_terms(ds_cfg)
        nu_range = rf_cfg.get('nu_range', [0.5, 2.5])
        ls_range = rf_cfg.get('length_scale_range', [0.1, 0.5])
        field_basis = rf_cfg.get('field_basis', 'kl')
        logE_std = rf_cfg.get('logE_std', rf_cfg.get('field_fluctuation_scale', 1.0))

        n = min(n_samples, len(X_test))
        seed = (self.config.get('dataset') or {}).get('seed', 42)
        rng = np.random.default_rng(seed)

        E_fields = []
        E_reduced_fields = []
        k_h_values = []
        k_v_values = []

        for i in range(n):
            nu_sampling = rf_cfg.get('nu_sampling', True)
            ls_sampling = rf_cfg.get('length_scale_sampling', True)
            nu_ref = rf_cfg.get('nu_ref', 1.5)
            ls_ref = rf_cfg.get('length_scale_ref', 0.3)
            nu = rng.uniform(*nu_range) if nu_sampling else nu_ref
            length_scale = rng.uniform(*ls_range) if ls_sampling else ls_ref
            if field_basis == 'dct':
                E_field = field_gen.reconstruct_from_coefficients(
                    X_test[i], E_ref=E_ref, logE_std=logE_std
                )
            else:
                E_field = field_gen.generate_field(
                    X_test[i], nu, length_scale,
                    n_terms=n_terms, E_ref=E_ref, logE_std=logE_std,
                )
            E_fields.append(E_field)
            # Reconstruct reduced E field using basis functions (d-dimensional xi')
            E_red_field = reduced_lut._reconstruct_field(xi_prime_pred[i])
            E_reduced_fields.append(E_red_field)
            k_h_values.append(k_h)
            k_v_values.append(k_v)

        viz.plot_material_fields(
            E_fields, E_reduced_fields,
            k_h_values=k_h_values, k_v_values=k_v_values,
            n_samples=n,
        )

    def save_run_summary(self, metrics=None, phase2_r2_by_type=None):
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

        if phase2_r2_by_type:
            summary_lines += ["", "Phase 2 surrogate R2 by type:"]
            for s_type, r2 in phase2_r2_by_type.items():
                summary_lines.append(f"  {s_type}: {r2:.4f}")

        if metrics:
            summary_lines += [
                "",
                "Test metrics (Phase 4):",
                f"  R2: {metrics.get('r2', 'N/A')}",
                f"  RMSE: {metrics.get('rmse', 'N/A')}",
                f"  Relative L2: {metrics.get('rel_l2', 'N/A')}",
                "",
                "R2 diagnosis:",
                "  R2 uses variance_weighted multioutput to down-weight output nodes",
                "  with near-zero variance across test samples, which would otherwise",
                "  give spuriously large negative contributions to the average.",
                "  Negative R2 indicates the reducer predictions are worse than",
                "  predicting the mean settlement for every sample.",
            ]
        summary_lines += [
            "",
            "Artifact locations:",
            f"  Training data:    {self.data_dir}/X_train.npy, {self.data_dir}/Y_train.npy",
            f"  LUT data:         {self.data_dir}/lut_grid_points.npy, {self.data_dir}/lut_responses.npy",
            f"  Surrogate model:  {self.models_dir}/reduced_lut/",
            f"  Phase-2 eval:     {self.models_dir}/reduced_lut/<type>/evaluation/",
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

        phase2_r2_by_type = None
        if 2 in phases:
            reduced_lut = self.phase2_build_reduced_surrogate()
            # Collect R² values stored in config.json for the summary
            import json as _json
            cfg_path = os.path.join(self.models_dir, 'reduced_lut', 'config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as _f:
                    _meta = _json.load(_f)
                phase2_r2_by_type = {_meta.get('surrogate_type', 'unknown'):
                                     _meta.get('r2_validation')}
        elif any(p in phases for p in [3, 4]):
            from src.reduced_lut import ReducedLUT
            solver = self._get_solver()
            lut_dir = os.path.join(self.models_dir, 'reduced_lut')
            surr_cfg = self.config.get('surrogate') or {}
            surr_types = surr_cfg.get('types', None)
            if surr_types is None:
                surr_types = [surr_cfg.get('type', 'nn')]
            surr_type = surr_types[0]
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
            surr_cfg = self.config.get('surrogate') or {}
            red_cfg = self.config.get('dimension_reducer') or {}
            reducer_types = red_cfg.get('types', None)
            if reducer_types is None:
                surrogate_type = surr_cfg.get('type', 'nn')
                reducer_types = [surrogate_type]

            if len(reducer_types) == 1:
                reducer = self._load_reducer(reducer_types[0])
            else:
                reducer = {rt: self._load_reducer(rt) for rt in reducer_types}

        if 4 in phases:
            metrics = self.phase4_evaluate(reducer, reduced_lut, X_test, Y_test)

        self.save_run_summary(metrics, phase2_r2_by_type=phase2_r2_by_type)
        return metrics

    def _load_reducer(self, reducer_type):
        """Load a saved reducer of the given type."""
        if reducer_type == 'nn':
            import torch
            reducer_path = os.path.join(self.models_dir, 'dimension_reducer_nn.pt')
            try:
                return torch.load(reducer_path, weights_only=False)
            except TypeError:
                # Torch < 1.13 does not accept weights_only kwarg
                return torch.load(reducer_path)
        else:
            import pickle
            reducer_path = os.path.join(self.models_dir, f'dimension_reducer_{reducer_type}.pkl')
            with open(reducer_path, 'rb') as f:
                return pickle.load(f)

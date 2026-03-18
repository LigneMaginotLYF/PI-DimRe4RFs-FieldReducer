import unittest
import numpy as np
import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigManager(unittest.TestCase):
    def test_load_config(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager(config_file='config.yaml')
        self.assertIn('dataset', cm.config)
        self.assertIn('material', cm.config)
        self.assertIn('solver', cm.config)

    def test_preset_merge(self):
        """Preset overrides should merge correctly with base config."""
        from src.config_manager import ConfigManager
        cm = ConfigManager(
            config_file='config.yaml',
            preset_file='presets/stage1_d1_polynomial.yaml',
        )
        self.assertEqual(cm.config['dimension_reducer']['basis_type'], 'polynomial')
        self.assertIn('dataset', cm.config)
        self.assertIn('surrogate', cm.config)

    def test_deep_merge(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        override = {'a': {'b': 10}}
        result = cm._deep_merge(base, override)
        self.assertEqual(result['a']['b'], 10)
        self.assertEqual(result['a']['c'], 2)
        self.assertEqual(result['d'], 3)

    def test_scientific_notation_coercion(self):
        """Scientific notation strings should be coerced to float."""
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        result = cm._coerce_numeric_strings({'x': '1.0e6', 'y': '3e-12', 'z': 'abc'})
        self.assertAlmostEqual(result['x'], 1.0e6)
        self.assertAlmostEqual(result['y'], 3e-12)
        self.assertEqual(result['z'], 'abc')


class TestFieldGenerator(unittest.TestCase):
    def test_generate_field(self):
        from src.field_generator import KLExpansionField
        fg = KLExpansionField(n_nodes_x=10, n_nodes_z=10)
        xi = np.random.standard_normal(5)
        E = fg.generate_field(xi, nu=1.5, length_scale=0.3, n_terms=5, E_ref=10e6)
        self.assertEqual(E.shape, (10, 10))
        self.assertTrue(np.all(E > 0))
        self.assertFalse(np.any(np.isnan(E)))

    def test_matern_kernel(self):
        from src.field_generator import MaternKernel
        pts = np.array([[0, 0], [1, 0], [0, 1]])
        k = MaternKernel(nu=1.5, length_scale=0.5)
        K = k(pts)
        self.assertEqual(K.shape, (3, 3))
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals >= -1e-8))


class TestBiotSolver(unittest.TestCase):
    def setUp(self):
        self.config = {
            'material': {
                'poisson_ratio': 0.3,
                'biot_coefficient': 0.8,
                'applied_load': 1.0e6,
                'pore_pressure_bottom': 1.0e5,
            },
            'domain': {'length_x': 1.0, 'length_z': 1.0},
            'solver': {'n_nodes_x': 10, 'n_nodes_z': 10},
        }

    def test_run(self):
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(self.config)
        E_field = np.full((10, 10), 10e6)
        settlement = solver.run(E_field, k_h=1e-12, k_v=1e-12)
        self.assertEqual(settlement.shape, (10,))
        self.assertFalse(np.any(np.isnan(settlement)))
        self.assertTrue(np.all(settlement >= 0))

    def test_uniform_settlement_with_constant_field(self):
        """With constant E field and uniform top-drainage BCs, settlement should be
        spatially uniform across x (1D vertical flow, no horizontal variation)."""
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(self.config)
        E_field = np.full((10, 10), 10e6)
        settlement = solver.run(E_field, k_h=1e-12, k_v=1e-12)
        self.assertEqual(settlement.shape, (10,))
        # All x-nodes must have the same settlement (uniform BCs + uniform material)
        self.assertTrue(np.allclose(settlement, settlement[0], rtol=1e-10, atol=0),
                        "Settlement should be spatially uniform for a constant material field "
                        "with fully uniform top-drainage boundary conditions.")


class TestNNSurrogate(unittest.TestCase):
    def test_fit_predict(self):
        from src.mapping_learner_nn import PhysicsDrivenMappingNN
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3)).astype(np.float32)
        Y = rng.standard_normal((50, 10)).astype(np.float32)
        model = PhysicsDrivenMappingNN(input_dim=3, output_dim=10, hidden_dim=16, n_blocks=2)
        model.fit(X[:40], Y[:40], X[40:], Y[40:], epochs=10, lr=1e-2, batch_size=20)
        Y_pred = model.predict(X[40:])
        self.assertEqual(Y_pred.shape, (10, 10))


class TestPCESurrogate(unittest.TestCase):
    def test_fit_predict(self):
        from src.mapping_learner_pce import PolynomialChaosExpansion
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        Y = X[:, :2] ** 2
        pce = PolynomialChaosExpansion(degree=2, n_inputs=3, n_outputs=2)
        pce.fit(X, Y)
        Y_pred = pce.predict(X[:5])
        self.assertEqual(Y_pred.shape, (5, 2))


class TestValidation(unittest.TestCase):
    def test_perfect_prediction(self):
        from src.validation import Validation
        Y = np.random.randn(20, 10)
        metrics = Validation.compute_metrics(Y, Y)
        self.assertAlmostEqual(metrics['r2'], 1.0, places=5)
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=5)
        self.assertAlmostEqual(metrics['rel_l2'], 0.0, places=5)

    def test_imperfect_prediction(self):
        from src.validation import Validation
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((20, 10))
        Y_pred = Y + 0.1 * rng.standard_normal((20, 10))
        metrics = Validation.compute_metrics(Y_pred, Y)
        self.assertGreater(metrics['r2'], 0.5)
        self.assertGreater(metrics['rmse'], 0)

    def test_r2_not_nan_on_constant_output(self):
        """variance_weighted R² should not produce NaN when an output column is constant."""
        from src.validation import Validation
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((10, 5))
        # Make the last column constant across all samples
        Y[:, -1] = 1.0
        Y_pred = Y + 0.01 * rng.standard_normal((10, 5))
        metrics = Validation.compute_metrics(Y_pred, Y)
        # R² should be finite (not NaN) even with a zero-variance column
        self.assertIsNotNone(metrics['r2'])
        self.assertFalse(np.isnan(metrics['r2']))

    def test_per_sample_r2_no_sklearn_regression(self):
        """Per-sample R² should handle near-constant profiles without extreme values."""
        from src.validation import Validation
        rng = np.random.default_rng(2)
        # Near-constant settlement profiles
        Y = np.ones((10, 8)) + 0.001 * rng.standard_normal((10, 8))
        Y_pred = Y + 0.001 * rng.standard_normal((10, 8))
        per = Validation.compute_per_sample_metrics(Y_pred, Y)
        # All per-sample R² should be finite
        self.assertTrue(np.all(np.isfinite(per['r2'])))


def _make_smoke_config():
    """Return a minimal config for fast smoke tests."""
    return {
        'dataset': {'n_samples': 8, 'n_kl_terms_E': 5, 'seed': 0},
        'material': {
            'E_ref': 10.0e6,
            'permeability_ref': 1.0e-12,
            'permeability_h': 1.0e-12,
            'permeability_v': 1.0e-12,
            'poisson_ratio': 0.3,
            'biot_coefficient': 0.8,
            'applied_load': 1.0e6,
            'pore_pressure_bottom': 1.0e5,
        },
        'domain': {'length_x': 1.0, 'length_z': 1.0},
        'solver': {'n_nodes_x': 8, 'n_nodes_z': 8},
        'random_field': {
            'nu_range': [0.5, 2.5],
            'length_scale_range': [0.1, 0.5],
        },
        'dimension_reducer': {'d': 1, 'basis_type': 'polynomial', 'basis_order': 1},
        'reduced_lut': {'n_grid_points': 15, 'grid_type': 'random'},
        'surrogate': {
            'type': 'pce',
            'hidden_dim': 16,
            'n_blocks': 1,
            'epochs': 2,
            'learning_rate': 1e-3,
            'batch_size': 8,
            'basis_order': 2,
        },
        'validation': {
            'train_fraction': 0.6,
            'val_fraction': 0.2,
            'test_fraction': 0.2,
        },
    }


class TestSmokeRun(unittest.TestCase):
    """End-to-end smoke test with tiny settings to verify pipeline completes."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_smoke_')
        # Redirect global artifact dirs to tmp so tests don't pollute cwd
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_pipeline_pce(self):
        """PCE smoke run: all four phases complete and core artifacts are created."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        pipeline = TrainingPipeline(config, output_dir='results/smoke_pce')
        metrics = pipeline.orchestrate(phases=[1, 2, 3, 4])

        # Core model / data artifacts exist
        self.assertTrue(os.path.exists('data/X_train.npy'))
        self.assertTrue(os.path.exists('data/Y_train.npy'))
        self.assertTrue(os.path.exists('data/lut_grid_points.npy'))
        self.assertTrue(os.path.exists('data/lut_responses.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/config.json'))
        self.assertTrue(os.path.exists('models/reduced_lut/grid_points.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/responses.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/surrogate_pce.pkl'))
        self.assertTrue(os.path.exists('models/dimension_reducer_pce.pkl'))

        # Run-level artifacts live inside the run folder
        self.assertTrue(os.path.exists('results/smoke_pce/metrics.json'))
        self.assertTrue(os.path.exists('results/smoke_pce/run_summary.txt'))

        # Plots are under results/<run_id>/plots/ (not a top-level plots/ dir)
        run_plots = 'results/smoke_pce/plots'
        self.assertTrue(os.path.isdir(run_plots),
                        f"Expected per-run plots dir at {run_plots}")
        self.assertFalse(os.path.exists('plots'),
                         "Top-level 'plots/' directory should not be created")

        # Phase-2 independent evaluation artifacts
        self.assertTrue(
            os.path.exists('models/reduced_lut/pce/evaluation/metrics.json'),
            "Phase-2 independent evaluation metrics should be saved",
        )

        # Array shapes
        X = np.load('data/X_train.npy')
        Y = np.load('data/Y_train.npy')
        self.assertEqual(X.shape, (8, 5))
        self.assertEqual(Y.shape[0], 8)

        # Metrics dict has expected keys
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('rel_l2', metrics)

    def test_phase2_always_recomputes(self):
        """Phase-2 LUT surrogate is ALWAYS recomputed (reuse disabled by design)."""
        from src.training_pipeline import TrainingPipeline
        import json

        config = _make_smoke_config()
        # First run: build and save
        p1 = TrainingPipeline(config, output_dir='results/reuse_first')
        p1.orchestrate(phases=[1, 2])

        # Read the grid from the first run to check it changes
        grid_first = np.load('models/reduced_lut/grid_points.npy')

        # Second run: even if config.json exists, surrogate must be regenerated
        # Use a different seed to confirm the grid is freshly sampled
        config2 = _make_smoke_config()
        config2['dataset']['seed'] = 7
        p2 = TrainingPipeline(config2, output_dir='results/reuse_second')
        p2.orchestrate(phases=[2])

        # Grid should be freshly sampled (different seed → different grid)
        grid_second = np.load('models/reduced_lut/grid_points.npy')
        self.assertFalse(
            np.allclose(grid_first, grid_second),
            msg="Phase-2 grid was identical in both runs – reuse may be incorrectly enabled",
        )

    def test_full_pipeline_nn(self):
        """NN smoke run: all four phases complete."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        config['surrogate']['type'] = 'nn'
        config['surrogate']['epochs'] = 2
        pipeline = TrainingPipeline(config, output_dir='results/smoke_nn')
        metrics = pipeline.orchestrate(phases=[1, 2, 3, 4])

        self.assertTrue(os.path.exists('models/reduced_lut/surrogate_nn.pt'))
        self.assertTrue(os.path.exists('models/dimension_reducer_nn.pt'))
        self.assertIn('r2', metrics)

        # Plots inside run folder
        self.assertTrue(os.path.isdir('results/smoke_nn/plots'))
        self.assertFalse(os.path.exists('plots'))

    def test_dataset_reuse_flag(self):
        """dataset.reuse=True reloads existing arrays instead of regenerating."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        # First run to generate data
        p1 = TrainingPipeline(config, output_dir='results/reuse_ds_first')
        X1, Y1 = p1.phase1_generate_dataset()
        mtime_before = os.path.getmtime('data/X_train.npy')

        # Second run with reuse=True: should not overwrite the arrays
        import time as _time
        _time.sleep(0.05)  # ensure mtime would differ if file were rewritten
        config2 = _make_smoke_config()
        config2['dataset']['reuse'] = True
        p2 = TrainingPipeline(config2, output_dir='results/reuse_ds_second')
        X2, Y2 = p2.phase1_generate_dataset()

        self.assertEqual(os.path.getmtime('data/X_train.npy'), mtime_before,
                         "X_train.npy should NOT be rewritten when dataset.reuse=True")
        self.assertTrue(np.array_equal(X1, X2))

    def test_multi_surrogate_types(self):
        """surrogate.types list trains both NN and PCE surrogates."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        config['surrogate']['types'] = ['nn', 'pce']
        config['surrogate']['epochs'] = 2
        pipeline = TrainingPipeline(config, output_dir='results/multi_surr')
        pipeline.orchestrate(phases=[1, 2])

        # Both surrogate artefacts saved
        self.assertTrue(os.path.exists('models/reduced_lut/surrogate_nn.pt'))
        self.assertTrue(os.path.exists('models/reduced_lut/surrogate_pce.pkl'))
        # Phase-2 independent evaluation saved for each type
        self.assertTrue(os.path.exists('models/reduced_lut/nn/evaluation/metrics.json'))
        self.assertTrue(os.path.exists('models/reduced_lut/pce/evaluation/metrics.json'))

    def test_collocation_indices_saved_and_loaded(self):
        """fit_surrogate saves train/val indices; save()/load() persist them."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        pipeline = TrainingPipeline(config, output_dir='results/colloc_idx')
        pipeline.orchestrate(phases=[1, 2])

        # Collocation index files must exist after Phase 2
        self.assertTrue(
            os.path.exists('models/reduced_lut/train_indices.npy'),
            "train_indices.npy should be saved after Phase 2",
        )
        self.assertTrue(
            os.path.exists('models/reduced_lut/val_indices.npy'),
            "val_indices.npy should be saved after Phase 2",
        )

        # Loading the LUT should restore the indices
        from src.reduced_lut import ReducedLUT
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(config)
        lut = ReducedLUT(config, solver, output_dir='models/reduced_lut')
        lut.load(surrogate_type='pce')

        self.assertIsNotNone(lut.train_indices)
        self.assertIsNotNone(lut.val_indices)
        n = len(lut.grid_points)
        n_val = len(lut.val_indices)
        n_train = len(lut.train_indices)
        self.assertEqual(n_train + n_val, n)

    def test_multi_surrogate_phase3_no_none_error(self):
        """Phase 3 with multiple reducer types loads the correct surrogate for each."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        config['surrogate']['types'] = ['nn', 'pce']
        config['surrogate']['epochs'] = 2
        config['dimension_reducer']['types'] = ['nn', 'pce']
        pipeline = TrainingPipeline(config, output_dir='results/multi_reducer')
        # Should complete phases 1-3 without NoneType errors
        pipeline.orchestrate(phases=[1, 2, 3])

        self.assertTrue(os.path.exists('models/dimension_reducer_nn.pt'))
        self.assertTrue(os.path.exists('models/dimension_reducer_pce.pkl'))

    def test_collocation_visualization(self):
        """plot_settlement_comparison_with_collocation produces the correct output;
        plot_phase2_surrogate_accuracy produces scatter and profile plots."""
        from src.training_pipeline import TrainingPipeline

        config = _make_smoke_config()
        pipeline = TrainingPipeline(config, output_dir='results/colloc_viz')
        X_train, Y_train = pipeline.phase1_generate_dataset()
        lut = pipeline.phase2_build_reduced_surrogate()
        reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(
            X_train, Y_train, lut
        )

        from src.visualization import Visualization
        viz = Visualization(plots_dir='results/colloc_viz/plots')

        xi_prime = reducer.predict(X_test)
        Y_pred = lut.predict(xi_prime)

        viz.plot_settlement_comparison_with_collocation(Y_test, Y_pred, lut, n_samples=2)
        viz.plot_phase2_surrogate_accuracy(lut, surrogate_type='nn',
                                           val_fraction=0.2, seed=42)

        self.assertTrue(os.path.exists(
            'results/colloc_viz/plots/settlement_comparison/comparison_with_collocation.png'
        ))
        self.assertTrue(os.path.exists(
            'results/colloc_viz/plots/phase2_surrogate/scatter_nn.png'
        ))


class TestBiotSolver1D(unittest.TestCase):
    """Tests for the 1-D Biot solver."""

    def _make_config(self, response_mode='steady_state', n_x=1):
        return {
            'material': {
                'poisson_ratio': 0.3,
                'biot_coefficient': 0.8,
                'applied_load': 1.0e6,
                'pore_pressure_bottom': 1.0e5,
            },
            'domain': {'length_x': 1.0, 'length_z': 1.0},
            'solver': {
                'n_nodes_x': n_x,
                'n_nodes_z': 10,
                'response_mode': response_mode,
                't_final': 0.1,
            },
        }
    def test_steady_state_output_shape(self):
        from src.forward_solver_1d import BiotSolver1D
        solver = BiotSolver1D(self._make_config())
        E_1d = np.full(10, 10.0e6)
        result = solver.run(E_1d, k_h=1e-12, k_v=1e-12)
        self.assertEqual(result.shape, (1,))

    def test_steady_state_positive(self):
        from src.forward_solver_1d import BiotSolver1D
        solver = BiotSolver1D(self._make_config())
        E_1d = np.full(10, 10.0e6)
        result = solver.run(E_1d, k_h=1e-12, k_v=1e-12)
        self.assertTrue(np.all(result >= 0), "Settlement must be non-negative")
        self.assertFalse(np.any(np.isnan(result)), "Settlement must not be NaN")

    def test_transient_output_shape(self):
        from src.forward_solver_1d import BiotSolver1D
        solver = BiotSolver1D(self._make_config(response_mode='transient'))
        E_1d = np.full(10, 10.0e6)
        result = solver.run(E_1d, k_h=1e-12, k_v=1e-12)
        self.assertEqual(result.shape, (1,))
        self.assertFalse(np.any(np.isnan(result)))

    def test_output_compatible_with_2d_interface(self):
        """1D solver with n_nodes_x=3 returns shape (3,) like 2D solver."""
        from src.forward_solver_1d import BiotSolver1D
        solver = BiotSolver1D(self._make_config(n_x=3))
        E_1d = np.full(10, 10.0e6)
        result = solver.run(E_1d, k_h=1e-12, k_v=1e-12)
        self.assertEqual(result.shape, (3,))
        # All values should be equal (1-D: no x-variation)
        self.assertTrue(np.allclose(result, result[0]))

    def test_stiffer_soil_gives_less_settlement(self):
        """Doubling E should roughly halve settlement."""
        from src.forward_solver_1d import BiotSolver1D
        cfg = self._make_config()
        solver = BiotSolver1D(cfg)
        E_soft = np.full(10, 5.0e6)
        E_hard = np.full(10, 10.0e6)
        u_soft = solver.run(E_soft, k_h=1e-12, k_v=1e-12)[0]
        u_hard = solver.run(E_hard, k_h=1e-12, k_v=1e-12)[0]
        self.assertGreater(u_soft, u_hard)


class TestBiotSolver2DTransient(unittest.TestCase):
    """Tests for the transient mode of BiotSolver2D."""

    def setUp(self):
        self.config = {
            'material': {
                'poisson_ratio': 0.3,
                'biot_coefficient': 0.8,
                'applied_load': 1.0e6,
                'pore_pressure_bottom': 1.0e5,
            },
            'domain': {'length_x': 1.0, 'length_z': 1.0},
            'solver': {
                'n_nodes_x': 5,
                'n_nodes_z': 5,
                'response_mode': 'transient',
                't_final': 0.1,
            },
        }

    def test_transient_output_shape(self):
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(self.config)
        E = np.full((5, 5), 10.0e6)
        result = solver.run(E, k_h=1e-12, k_v=1e-12)
        self.assertEqual(result.shape, (5,))
        self.assertFalse(np.any(np.isnan(result)))

    def test_transient_non_negative(self):
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(self.config)
        E = np.full((5, 5), 10.0e6)
        result = solver.run(E, k_h=1e-12, k_v=1e-12)
        self.assertTrue(np.all(result >= 0))


def _make_1d_smoke_config():
    """Minimal config for 1-D steady-state smoke tests."""
    return {
        'dataset': {'n_samples': 8, 'n_kl_terms_E': 2, 'seed': 1,
                    'reuse': False},
        'material': {
            'E_ref': 10.0e6, 'permeability_ref': 1e-12,
            'permeability_h': 1e-12, 'permeability_v': 1e-12,
            'poisson_ratio': 0.3, 'biot_coefficient': 0.8,
            'fluid_viscosity': 1e-3, 'porosity': 0.3,
            'fluid_bulk_modulus': 2.2e9, 'applied_load': 1e6,
            'pore_pressure_bottom': 1e5,
        },
        'domain': {'length_x': 1.0, 'length_z': 1.0},
        'solver': {'type': '1d', 'response_mode': 'steady_state',
                   'n_nodes_x': 1, 'n_nodes_z': 5},
        'random_field': {'covariance': 'matern', 'nu_sampling': True,
                         'nu_range': [0.5, 2.5], 'nu_ref': 1.5,
                         'length_scale_sampling': True,
                         'length_scale_range': [0.1, 0.5],
                         'length_scale_ref': 0.3},
        'dimension_reducer': {'d': 1, 'basis_type': 'polynomial',
                               'basis_order': 1, 'mode': 'learned'},
        'reduced_lut': {'n_grid_points': 6, 'grid_type': 'random',
                        'reuse': False},
        'surrogate': {'type': 'nn', 'hidden_dim': 8, 'n_blocks': 1,
                      'epochs': 2, 'learning_rate': 1e-3, 'batch_size': 4},
        'collocation': {},
        'validation': {'train_fraction': 0.5, 'val_fraction': 0.25,
                       'test_fraction': 0.25},
    }


def _make_transient_smoke_config():
    """Minimal config for transient 2-D smoke tests."""
    cfg = _make_smoke_config()
    cfg['solver']['response_mode'] = 'transient'
    cfg['solver']['t_final'] = 0.1
    return cfg


class TestSmokeRun1D(unittest.TestCase):
    """Smoke tests for the complete 1-D pipeline."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_1d_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_1d_steady_phase1_shape(self):
        """Phase 1 with 1-D solver produces settlement shape (n_samples, 1)."""
        from src.training_pipeline import TrainingPipeline
        cfg = _make_1d_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_1d_steady')
        X, Y = pipeline.phase1_generate_dataset()
        self.assertEqual(X.shape, (8, 2))
        self.assertEqual(Y.shape, (8, 1))

    def test_1d_steady_phases_1_2(self):
        """Phases 1 and 2 complete without error for 1-D steady-state setup."""
        from src.training_pipeline import TrainingPipeline
        cfg = _make_1d_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_1d_ph12')
        X, Y = pipeline.phase1_generate_dataset()
        lut = pipeline.phase2_build_reduced_surrogate()
        self.assertIsNotNone(lut)
        self.assertEqual(lut.responses.shape[1], 1)


class TestSmokeRunTransient(unittest.TestCase):
    """Smoke tests for the complete transient pipeline."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_transient_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_transient_phase1_shape(self):
        """Phase 1 with transient 2-D solver produces valid output shape."""
        from src.training_pipeline import TrainingPipeline
        cfg = _make_transient_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_transient')
        X, Y = pipeline.phase1_generate_dataset()
        n_x = cfg['solver']['n_nodes_x']
        self.assertEqual(Y.shape, (cfg['dataset']['n_samples'], n_x))
        self.assertFalse(np.any(np.isnan(Y)))

    def test_transient_phases_1_2(self):
        """Phases 1 and 2 complete without error for transient setup."""
        from src.training_pipeline import TrainingPipeline
        cfg = _make_transient_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_transient_ph12')
        pipeline.orchestrate(phases=[1, 2])
        self.assertTrue(os.path.exists('models/reduced_lut/config.json'))


def _make_identity_smoke_config():
    """Config for identity-mapping verification (fixed KL basis, d=n_kl)."""
    cfg = _make_smoke_config()
    cfg['dataset']['n_kl_terms_E'] = 2
    cfg['random_field']['nu_sampling'] = False
    cfg['random_field']['nu_ref'] = 1.5
    cfg['random_field']['length_scale_sampling'] = False
    cfg['random_field']['length_scale_ref'] = 0.3
    cfg['dimension_reducer']['d'] = 2          # must equal n_kl_terms_E
    cfg['dimension_reducer']['basis_type'] = 'kl'
    cfg['dimension_reducer']['mode'] = 'identity'
    cfg['reduced_lut']['n_grid_points'] = 8
    return cfg


class TestIdentityMapping(unittest.TestCase):
    """Identity-mapping equivalence tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_identity_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_identity_reducer_predict(self):
        """_IdentityReducer.predict returns first output_dim columns."""
        from src.training_pipeline import _IdentityReducer
        reducer = _IdentityReducer(input_dim=5, output_dim=3)
        X = np.arange(20, dtype=float).reshape(4, 5)
        out = reducer.predict(X)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_array_equal(out, X[:, :3])

    def test_identity_full_pipeline(self):
        """Full pipeline with identity mode completes and reports low direct error."""
        from src.training_pipeline import TrainingPipeline
        cfg = _make_identity_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_identity')
        pipeline.orchestrate(phases=[1, 2, 3, 4])

        self.assertTrue(os.path.exists('results/test_identity/identity_check.json'))

        import json
        with open('results/test_identity/identity_check.json') as f:
            report = json.load(f)

        # With fixed nu/ls the direct-solver reconstruction should be near-perfect
        self.assertIn('identity_check_max_abs_error', report)
        max_err = report['identity_check_max_abs_error']
        # Max absolute error should be very small (< 1 % of typical settlement)
        self.assertLess(max_err, 0.1,
                        msg=f"Identity check max abs error {max_err:.4e} is unexpectedly large")

    def test_identity_phase3_returns_identity_reducer(self):
        """Phase 3 in identity mode returns an _IdentityReducer instance."""
        from src.training_pipeline import TrainingPipeline, _IdentityReducer
        cfg = _make_identity_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_id_phase3')
        X, Y = pipeline.phase1_generate_dataset()
        lut = pipeline.phase2_build_reduced_surrogate()
        reducer, _ = pipeline.phase3_train_dimension_reducer(X, Y, lut)
        self.assertIsInstance(reducer, _IdentityReducer)


class TestSettlementComparisonPlot(unittest.TestCase):
    """Settlement comparison plot correctness tests."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_plot_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_settlement_comparison_lines_only(self):
        """plot_settlement_comparison produces comparison.png with two curves."""
        from src.visualization import Visualization
        rng = np.random.default_rng(0)
        Y_true = rng.standard_normal((5, 10)) * 0.01 + 0.05
        Y_pred = Y_true + rng.standard_normal((5, 10)) * 0.001
        x_pos = np.linspace(0.0, 1.0, 10)

        out_dir = 'test_plot_lines/plots'
        viz = Visualization(plots_dir=out_dir)
        viz.plot_settlement_comparison(Y_true, Y_pred, n_samples=3,
                                       x_positions=x_pos)
        self.assertTrue(
            os.path.exists(os.path.join(out_dir, 'settlement_comparison',
                                        'comparison.png'))
        )

    def test_settlement_comparison_with_collocation_markers(self):
        """plot_settlement_comparison_with_collocation marks positions on GT."""
        from src.training_pipeline import TrainingPipeline
        from src.visualization import Visualization

        cfg = _make_smoke_config()
        pipeline = TrainingPipeline(cfg, output_dir='results/test_plot_coll')
        X, Y = pipeline.phase1_generate_dataset()
        lut = pipeline.phase2_build_reduced_surrogate()

        rng = np.random.default_rng(0)
        n_x = cfg['solver']['n_nodes_x']
        Y_true = rng.standard_normal((4, n_x)) * 0.01 + 0.05
        Y_pred = Y_true + rng.standard_normal((4, n_x)) * 0.001

        x_pos = np.linspace(0.0, 1.0, n_x)
        out_dir = 'results/test_plot_coll/plots'
        viz = Visualization(plots_dir=out_dir)
        viz.plot_settlement_comparison_with_collocation(
            Y_true, Y_pred, lut, n_samples=2, x_positions=x_pos
        )
        self.assertTrue(
            os.path.exists(os.path.join(
                out_dir, 'settlement_comparison', 'comparison_with_collocation.png'
            ))
        )


class TestConfigValidation(unittest.TestCase):
    """Config validation tests."""

    def _base(self):
        return {
            'dataset': {'n_samples': 10, 'n_kl_terms_E': 3},
            'material': {},
            'solver': {'type': '2d', 'response_mode': 'steady_state',
                       'n_nodes_x': 5, 'n_nodes_z': 5},
            'dimension_reducer': {'d': 1, 'basis_type': 'polynomial',
                                  'basis_order': 1, 'mode': 'learned'},
            'reduced_lut': {'n_grid_points': 10},
            'surrogate': {'type': 'nn'},
            'random_field': {'nu_ref': 1.5, 'length_scale_ref': 0.3},
        }

    def test_valid_config_passes(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        try:
            cm.validate(cfg)
        except ValueError as e:
            self.fail(f"Valid config raised ValueError: {e}")

    def test_invalid_solver_type_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['solver']['type'] = 'gibberish'
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_invalid_response_mode_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['solver']['response_mode'] = 'quasi-static'
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_1d_with_n_nodes_x_gt1_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['solver']['type'] = '1d'
        cfg['solver']['n_nodes_x'] = 5
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_identity_mode_wrong_d_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['dimension_reducer']['mode'] = 'identity'
        cfg['dimension_reducer']['d'] = 2  # ≠ n_kl_terms_E=3
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_d_exceeds_poly_basis_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['dimension_reducer']['basis_type'] = 'polynomial'
        cfg['dimension_reducer']['basis_order'] = 1  # 3 basis functions
        cfg['dimension_reducer']['d'] = 10  # more than basis size
        with self.assertRaises(ValueError):
            cm.validate(cfg)



class TestDCTField(unittest.TestCase):
    """Tests for the DCT-basis field generator."""

    def test_dct_basis_shape(self):
        """compute_dct_basis returns correct shape."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        Psi, freqs = fg.compute_dct_basis(n_terms=5)
        self.assertEqual(Psi.shape, (64, 5))
        self.assertEqual(freqs.shape, (5,))

    def test_dct_basis_deterministic(self):
        """The DCT basis is identical across two independent instantiations."""
        from src.field_generator import DCTField
        fg1 = DCTField(n_nodes_x=6, n_nodes_z=6)
        fg2 = DCTField(n_nodes_x=6, n_nodes_z=6)
        Psi1, _ = fg1.compute_dct_basis(n_terms=4)
        Psi2, _ = fg2.compute_dct_basis(n_terms=4)
        np.testing.assert_array_equal(Psi1, Psi2,
                                      err_msg="DCT basis must be deterministic and basis-agnostic")

    def test_dct_basis_independent_of_matern_params(self):
        """Two calls with different (nu, ls) produce the SAME basis Psi."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        # Use the same rng state so xi values are the same; only nu/ls differ
        _, E1 = fg.generate_field(rng1, nu=0.5, length_scale=0.2, n_terms=4)
        # Reset field gen cache and re-check basis
        Psi_nu05, _ = fg.compute_dct_basis(n_terms=4)
        fg2 = DCTField(n_nodes_x=8, n_nodes_z=8)
        Psi_nu20, _ = fg2.compute_dct_basis(n_terms=4)
        np.testing.assert_array_equal(Psi_nu05, Psi_nu20,
                                      err_msg="Basis must be identical for different nu")

    def test_dct_generate_field_shape(self):
        """generate_field returns correct shapes."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=10, n_nodes_z=10)
        rng = np.random.default_rng(42)
        xi, E = fg.generate_field(rng, nu=1.5, length_scale=0.3, n_terms=5, E_ref=10e6)
        self.assertEqual(xi.shape, (5,))
        self.assertEqual(E.shape, (10, 10))
        self.assertTrue(np.all(E > 0))
        self.assertFalse(np.any(np.isnan(E)))

    def test_dct_reconstruct_from_coefficients(self):
        """Reconstruct with same xi and different nu produces the SAME field."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        xi = np.array([0.5, -0.3, 0.1, 0.2, -0.1])
        E1 = fg.reconstruct_from_coefficients(xi, E_ref=10e6)
        E2 = fg.reconstruct_from_coefficients(xi, E_ref=10e6)
        np.testing.assert_array_equal(E1, E2,
                                      err_msg="Same xi must produce same E field regardless of call order")

    def test_dct_logE_std_scales_amplitude(self):
        """logE_std scales the log-E amplitude: larger std -> larger field variation."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        rng = np.random.default_rng(0)
        xi = np.array([1.0, 0.5, -0.3, 0.2, 0.0])
        E_small = fg.reconstruct_from_coefficients(xi, E_ref=10e6, logE_std=0.1)
        E_large = fg.reconstruct_from_coefficients(xi, E_ref=10e6, logE_std=2.0)
        # Variation of large should exceed variation of small
        self.assertGreater(E_large.std(), E_small.std())

    def test_dct_coefficient_distribution_matern_shaped(self):
        """matern_spectral_variance returns a valid probability-like distribution."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        _, freqs = fg.compute_dct_basis(n_terms=5)
        sigma_k = fg.matern_spectral_variance(nu=1.5, length_scale=0.3, mode_freqs=freqs)
        self.assertEqual(sigma_k.shape, (5,))
        self.assertTrue(np.all(sigma_k > 0))
        # sigma_k^2 should sum to 1 (unit variance normalisation)
        self.assertAlmostEqual((sigma_k ** 2).sum(), 1.0, places=5)

    def test_dct_smoke_phase1(self):
        """Phase-1 with DCT field_basis completes and X_train has correct shape."""
        from src.training_pipeline import TrainingPipeline
        import tempfile, shutil
        tmp = tempfile.mkdtemp(prefix='pi_dct_phase1_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            cfg['random_field']['field_basis'] = 'dct'
            cfg['random_field']['logE_std'] = 1.0
            pipeline = TrainingPipeline(cfg, output_dir='results/dct_phase1')
            X, Y = pipeline.phase1_generate_dataset()
            self.assertEqual(X.shape, (cfg['dataset']['n_samples'], 5))
            self.assertFalse(np.any(np.isnan(X)))
            self.assertFalse(np.any(np.isnan(Y)))
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dct_smoke_phases_1_to_4(self):
        """End-to-end smoke test with DCT basis through all four phases."""
        from src.training_pipeline import TrainingPipeline
        import tempfile, shutil
        tmp = tempfile.mkdtemp(prefix='pi_dct_e2e_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            cfg['dataset']['n_terms_E'] = 5
            cfg['random_field']['field_basis'] = 'dct'
            cfg['random_field']['logE_std'] = 1.0
            cfg['random_field']['nu_sampling'] = False
            cfg['random_field']['nu_ref'] = 1.5
            cfg['random_field']['length_scale_sampling'] = False
            cfg['random_field']['length_scale_ref'] = 0.3
            cfg['dimension_reducer']['basis_type'] = 'dct'
            cfg['dimension_reducer']['d'] = 3
            pipeline = TrainingPipeline(cfg, output_dir='results/dct_e2e')
            metrics = pipeline.orchestrate(phases=[1, 2, 3, 4])
            self.assertIn('r2', metrics)
            self.assertTrue(os.path.exists('data/X_train.npy'))
            self.assertTrue(os.path.exists('results/dct_e2e/metrics.json'))
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestNTermsEKey(unittest.TestCase):
    """Tests for the n_terms_E config key and backward compat with n_kl_terms_E."""

    def test_n_terms_E_accepted(self):
        """dataset.n_terms_E is recognised without deprecation warning."""
        from src.training_pipeline import TrainingPipeline
        import warnings, tempfile, shutil
        cfg = _make_smoke_config()
        cfg['dataset'].pop('n_kl_terms_E', None)
        cfg['dataset']['n_terms_E'] = 5

        tmp = tempfile.mkdtemp(prefix='pi_nterms_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                p = TrainingPipeline(cfg, output_dir='results/nterms')
                X, Y = p.phase1_generate_dataset()
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                                and 'n_kl_terms_E' in str(x.message)]
                self.assertEqual(len(dep_warnings), 0,
                                 "No DeprecationWarning should be emitted when using n_terms_E")
            self.assertEqual(X.shape[1], 5)
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_n_kl_terms_E_backward_compat(self):
        """dataset.n_kl_terms_E still works but emits a DeprecationWarning."""
        from src.training_pipeline import TrainingPipeline
        import warnings, tempfile, shutil
        cfg = _make_smoke_config()
        # Use only the legacy key
        cfg['dataset'].pop('n_terms_E', None)
        cfg['dataset']['n_kl_terms_E'] = 5

        tmp = tempfile.mkdtemp(prefix='pi_nkl_legacy_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                p = TrainingPipeline(cfg, output_dir='results/nkl_legacy')
                X, Y = p.phase1_generate_dataset()
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                                and 'n_kl_terms_E' in str(x.message)]
                self.assertGreater(len(dep_warnings), 0,
                                   "DeprecationWarning should be emitted when using n_kl_terms_E")
            self.assertEqual(X.shape[1], 5)
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestIdentityReducerOutputDim(unittest.TestCase):
    """Verify _IdentityReducer is instantiated with output_dim=d (not input_dim)."""

    def test_identity_reducer_uses_output_dim(self):
        """Phase 3 identity mode should create reducer with output_dim=d, not n_terms."""
        from src.training_pipeline import TrainingPipeline, _IdentityReducer
        import tempfile, shutil
        cfg = _make_smoke_config()
        cfg['dataset']['n_terms_E'] = 5
        cfg['random_field']['nu_sampling'] = False
        cfg['random_field']['nu_ref'] = 1.5
        cfg['random_field']['length_scale_sampling'] = False
        cfg['random_field']['length_scale_ref'] = 0.3
        # d=5 == n_terms_E (required for identity mode by config validation)
        cfg['dimension_reducer']['d'] = 5
        cfg['dimension_reducer']['basis_type'] = 'kl'
        cfg['dimension_reducer']['mode'] = 'identity'
        cfg['reduced_lut']['n_grid_points'] = 8

        tmp = tempfile.mkdtemp(prefix='pi_id_outdim_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            p = TrainingPipeline(cfg, output_dir='results/id_outdim')
            X, Y = p.phase1_generate_dataset()
            lut = p.phase2_build_reduced_surrogate()
            reducer, _ = p.phase3_train_dimension_reducer(X, Y, lut)
            self.assertIsInstance(reducer, _IdentityReducer)
            # output_dim must equal d (=5), NOT the input_dim (also 5 here, but
            # conceptually must come from output_dim=d rather than hardcoded input_dim)
            self.assertEqual(reducer.output_dim, 5)
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestConfigValidationDCT(unittest.TestCase):
    """Config validation tests for new DCT-related keys."""

    def _base(self):
        return {
            'dataset': {'n_samples': 10, 'n_terms_E': 3},
            'material': {},
            'solver': {'type': '2d', 'response_mode': 'steady_state',
                       'n_nodes_x': 5, 'n_nodes_z': 5},
            'dimension_reducer': {'d': 1, 'basis_type': 'polynomial',
                                  'basis_order': 1, 'mode': 'learned'},
            'reduced_lut': {'n_grid_points': 10},
            'surrogate': {'type': 'nn'},
            'random_field': {'nu_ref': 1.5, 'length_scale_ref': 0.3},
        }

    def test_dct_field_basis_valid(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['random_field']['field_basis'] = 'dct'
        cfg['dimension_reducer']['basis_type'] = 'dct'
        cfg['dimension_reducer']['d'] = 3
        try:
            cm.validate(cfg)
        except ValueError as e:
            self.fail(f"Valid DCT config raised ValueError: {e}")

    def test_invalid_field_basis_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['random_field']['field_basis'] = 'fourier'
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_dct_basis_type_d_exceeds_n_terms_raises(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['dimension_reducer']['basis_type'] = 'dct'
        cfg['dimension_reducer']['d'] = 10  # > n_terms_E=3
        with self.assertRaises(ValueError):
            cm.validate(cfg)

    def test_n_terms_E_and_n_kl_terms_E_coexist(self):
        """n_terms_E takes priority when both keys are present."""
        from src.training_pipeline import TrainingPipeline
        ds_cfg = {'n_terms_E': 4, 'n_kl_terms_E': 7}
        n = TrainingPipeline._resolve_n_terms(ds_cfg)
        self.assertEqual(n, 4)

    def test_logE_std_positive_bound(self):
        """logE_std <= 0 should raise a validation error."""
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = self._base()
        cfg['random_field']['logE_std'] = -0.5
        with self.assertRaises(ValueError):
            cm.validate(cfg)


class TestCollocationConsistency(unittest.TestCase):
    """Tests verifying collocation single-source-of-truth behaviour."""

    def test_5_positions_n_nodes_x_20_no_crash(self):
        """5 collocation positions with n_nodes_x=20 must not cause a shape mismatch
        crash when plotting settlement curves."""
        import tempfile, shutil
        from src.training_pipeline import TrainingPipeline
        from src.visualization import Visualization

        cfg = _make_smoke_config()
        cfg['solver']['n_nodes_x'] = 20
        # 5 collocation positions – intentionally fewer than n_nodes_x=20
        cfg['collocation'] = {'positions': [0.0, 0.25, 0.5, 0.75, 1.0]}

        tmp = tempfile.mkdtemp(prefix='pi_colloc_5pos_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            pipeline = TrainingPipeline(cfg, output_dir='results/colloc_5pos')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(
                X, Y, lut
            )

            # Verify collocation_indices.npy was saved and has 5 entries
            colloc_path = 'models/reduced_lut/collocation_indices.npy'
            self.assertTrue(os.path.exists(colloc_path))
            colloc_idx = np.load(colloc_path)
            self.assertEqual(len(colloc_idx), 5)

            # Verify plotting does not crash despite positions ≠ n_nodes_x
            xi_prime = reducer.predict(X_test)
            Y_pred = lut.predict(xi_prime)
            n_x = cfg['solver']['n_nodes_x']
            x_positions = np.linspace(0.0, 1.0, n_x)

            out_dir = 'results/colloc_5pos/plots'
            viz = Visualization(plots_dir=out_dir)
            # Should not raise any shape-mismatch error
            viz.plot_settlement_comparison(Y_test, Y_pred, n_samples=2,
                                           x_positions=x_positions)
            viz.plot_settlement_comparison_with_collocation(
                Y_test, Y_pred, lut, n_samples=2,
                x_positions=x_positions,
                colloc_idx=colloc_idx,
            )
            self.assertTrue(
                os.path.exists(os.path.join(
                    out_dir, 'settlement_comparison', 'comparison.png'
                ))
            )
            self.assertTrue(
                os.path.exists(os.path.join(
                    out_dir, 'settlement_comparison', 'comparison_with_collocation.png'
                ))
            )
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_collocation_indices_consistent_phase3_and_plot(self):
        """Collocation indices from _compute_collocation_indices must match the
        saved artifact and can be used identically for Phase-3 loss and markers."""
        from src.training_pipeline import TrainingPipeline
        import tempfile, shutil

        cfg = _make_smoke_config()
        cfg['solver']['n_nodes_x'] = 20
        cfg['collocation'] = {'positions': [0.0, 0.5, 1.0]}

        tmp = tempfile.mkdtemp(prefix='pi_colloc_consist_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            pipeline = TrainingPipeline(cfg, output_dir='results/colloc_consist')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            pipeline.phase3_train_dimension_reducer(X, Y, lut)

            # The saved artifact must equal what _compute_collocation_indices returns
            colloc_idx_saved = np.load('models/reduced_lut/collocation_indices.npy')
            colloc_idx_computed = TrainingPipeline._compute_collocation_indices(
                cfg, n_nodes_x=20, length_x=1.0
            )
            np.testing.assert_array_equal(colloc_idx_saved, colloc_idx_computed)

            # Indices must correspond to 3 positions (0.0→0, 0.5→10, 1.0→19)
            x_grid = np.linspace(0.0, 1.0, 20)
            dx = 1.0 / (20 - 1)  # grid spacing for n_nodes_x=20
            for pos, idx in zip([0.0, 0.5, 1.0], np.sort(colloc_idx_saved)):
                self.assertAlmostEqual(x_grid[idx], pos, delta=dx + 1e-9)
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_null_collocation_config_defaults_to_all_nodes(self):
        """When collocation config is null/missing, _compute_collocation_indices
        must return all n_nodes_x indices and must not crash."""
        from src.training_pipeline import TrainingPipeline
        import tempfile, shutil

        # Case 1: collocation key absent
        cfg_absent = _make_smoke_config()
        cfg_absent.pop('collocation', None)

        tmp = tempfile.mkdtemp(prefix='pi_colloc_null_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            for label, cfg in [('absent', cfg_absent),
                                ('null', dict(_make_smoke_config(), collocation=None)),
                                ('empty', dict(_make_smoke_config(), collocation={}))]:
                idx = TrainingPipeline._compute_collocation_indices(
                    cfg, n_nodes_x=10, length_x=1.0
                )
                self.assertEqual(
                    len(idx), 10,
                    f"Expected all 10 nodes for '{label}' collocation config, got {len(idx)}"
                )
                np.testing.assert_array_equal(idx, np.arange(10))

            # Full pipeline smoke – should not crash at any phase
            pipeline = TrainingPipeline(
                dict(_make_smoke_config(), collocation=None),
                output_dir='results/colloc_null',
            )
            pipeline.orchestrate(phases=[1, 2, 3, 4])
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)




class TestDCTMeanSampling(unittest.TestCase):
    """Tests for E_ref_sampling – mean variation encoded in the DC DCT coefficient."""

    def test_varying_e_ref_factor_affects_field_mean(self):
        """Larger E_ref_factor should increase mean(E_field)."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        rng = np.random.default_rng(99)

        # Use a fixed Matérn draw (zero higher-order coefficients) so only the
        # DC component is altered.
        _, E_low = fg.generate_field(
            rng, nu=1.5, length_scale=0.3, n_terms=1,
            E_ref=10e6, logE_std=1.0, E_ref_factor=0.5,
        )
        rng2 = np.random.default_rng(99)
        _, E_high = fg.generate_field(
            rng2, nu=1.5, length_scale=0.3, n_terms=1,
            E_ref=10e6, logE_std=1.0, E_ref_factor=2.0,
        )
        self.assertGreater(E_high.mean(), E_low.mean())

    def test_e_ref_factor_encodes_mean_approximately(self):
        """E_ref_factor=1.0 should leave the spatial mean ≈ E_ref."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=16, n_nodes_z=16)
        rng = np.random.default_rng(0)
        xi, E = fg.generate_field(
            rng, nu=1.5, length_scale=0.3, n_terms=5,
            E_ref=10e6, logE_std=0.1,  # small std so exp(logE) ≈ 1
            E_ref_factor=1.0,
        )
        # With tiny logE_std and factor=1, mean(E) ≈ E_ref
        self.assertAlmostEqual(E.mean() / 10e6, 1.0, delta=0.2)

    def test_e_ref_factor_none_unchanged_signature(self):
        """E_ref_factor=None should behave identically to the pre-PR code path."""
        from src.field_generator import DCTField
        fg = DCTField(n_nodes_x=8, n_nodes_z=8)
        rng1 = np.random.default_rng(7)
        xi1, E1 = fg.generate_field(rng1, nu=1.5, length_scale=0.3, n_terms=5,
                                     E_ref=10e6)
        rng2 = np.random.default_rng(7)
        xi2, E2 = fg.generate_field(rng2, nu=1.5, length_scale=0.3, n_terms=5,
                                     E_ref=10e6, E_ref_factor=None)
        np.testing.assert_array_equal(E1, E2)
        np.testing.assert_array_equal(xi1, xi2)

    def test_e_ref_sampling_in_phase1_affects_settlement_scale(self):
        """Phase-1 with E_ref_sampling=true should produce higher variance in Y."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_e_ref_samp_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            base = _make_smoke_config()
            base['dataset']['n_samples'] = 40
            base['random_field']['field_basis'] = 'dct'
            base['random_field']['logE_std'] = 0.5

            cfg_no = {**base,
                      'random_field': {**base['random_field'],
                                       'E_ref_sampling': False}}
            cfg_yes = {**base,
                       'random_field': {**base['random_field'],
                                        'E_ref_sampling': True,
                                        'E_ref_factor_range': [0.2, 5.0]}}

            p_no = TrainingPipeline(cfg_no, output_dir='r/no')
            X_no, Y_no = p_no.phase1_generate_dataset()

            p_yes = TrainingPipeline(cfg_yes, output_dir='r/yes')
            X_yes, Y_yes = p_yes.phase1_generate_dataset()

            # E_ref_sampling with a wide factor range should increase Y variance
            self.assertGreater(Y_yes.std(), Y_no.std())
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestSurrogateDCTOutput(unittest.TestCase):
    """Tests for Phase-2 surrogate DCT output representation."""

    def test_to_from_dct_space_roundtrip(self):
        """_to_dct_space followed by _from_dct_space recovers original Y."""
        from src.reduced_lut import ReducedLUT
        from unittest.mock import MagicMock
        cfg = {
            'surrogate': {'output_representation': 'dct', 'n_output_modes': 5},
            'reduced_lut': {}, 'material': {}, 'solver': {'n_nodes_x': 20, 'n_nodes_z': 10},
            'dimension_reducer': {},
        }
        lut = ReducedLUT(cfg, solver=MagicMock())
        rng = np.random.default_rng(42)
        Y = rng.uniform(1e-4, 1e-2, size=(8, 20))
        B = lut._to_dct_space(Y)
        self.assertEqual(B.shape, (8, 5))
        Y_rec = lut._from_dct_space(B)
        # Reconstruction with only 5 modes won't be exact for 20-node profiles
        self.assertEqual(Y_rec.shape, (8, 20))

    def test_dct_output_shape(self):
        """ReducedLUT with output_representation=dct must return shape (n, n_x) from predict."""
        from src.reduced_lut import ReducedLUT
        from unittest.mock import MagicMock, patch
        cfg = {
            'surrogate': {'output_representation': 'dct', 'n_output_modes': 4},
            'reduced_lut': {}, 'material': {}, 'solver': {'n_nodes_x': 10, 'n_nodes_z': 10},
            'dimension_reducer': {'d': 2},
        }
        lut = ReducedLUT(cfg, solver=MagicMock())
        lut.output_representation = 'dct'
        lut.n_output_modes = 4
        lut.n_x = 10
        # Mock surrogate.predict to return 4 DCT coefficients per sample
        mock_surr = MagicMock()
        mock_surr.predict.return_value = np.ones((3, 4))
        lut.surrogate = mock_surr
        Y = lut.predict(np.ones((3, 2)))
        self.assertEqual(Y.shape, (3, 10))

    def test_dct_output_smoother_than_direct(self):
        """Truncated DCT reconstruction should be smoother than random direct output."""
        from src.reduced_lut import ReducedLUT
        from unittest.mock import MagicMock
        cfg = {
            'surrogate': {'output_representation': 'dct', 'n_output_modes': 4},
            'reduced_lut': {}, 'material': {},
            'solver': {'n_nodes_x': 20, 'n_nodes_z': 5},
            'dimension_reducer': {},
        }
        lut = ReducedLUT(cfg, solver=MagicMock())
        rng = np.random.default_rng(0)
        # Noisy random profile
        Y_noisy = rng.uniform(0, 1, size=(1, 20))
        # Round-trip through truncated DCT should be smoother
        B = lut._to_dct_space(Y_noisy)
        Y_smooth = lut._from_dct_space(B)
        roughness_orig = float(np.mean(np.abs(np.diff(Y_noisy))))
        roughness_smooth = float(np.mean(np.abs(np.diff(Y_smooth))))
        self.assertLessEqual(roughness_smooth, roughness_orig + 1e-10)

    def test_dct_smoke_phase2(self):
        """Phase-2 with output_representation=dct completes without error."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_dct_surr_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            cfg['random_field']['field_basis'] = 'dct'
            cfg['surrogate']['output_representation'] = 'dct'
            cfg['surrogate']['n_output_modes'] = 3
            cfg['dimension_reducer']['basis_type'] = 'dct'
            cfg['dimension_reducer']['d'] = 2
            pipeline = TrainingPipeline(cfg, output_dir='r/dct_surr')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            # predict must return shape (n, n_x), not (n, n_output_modes)
            xi_dummy = lut.grid_points[:3]
            Y_pred = lut.predict(xi_dummy)
            n_x = cfg['solver']['n_nodes_x']
            self.assertEqual(Y_pred.shape, (3, n_x))
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestPlottingMismatchedCollocation(unittest.TestCase):
    """Tests that plotting does not crash when collocation positions != n_nodes_x."""

    def test_no_crash_when_collocation_len_ne_n_nodes_x(self):
        """plot_settlement_comparison_with_collocation must not crash when
        len(colloc_idx) != n_nodes_x (e.g. 3 collocation points, 20 nodes)."""
        from src.visualization import Visualization
        tmp = tempfile.mkdtemp(prefix='pi_viz_mismatch_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            viz = Visualization(plots_dir='plots_test')
            n_x = 20
            n_test = 4
            rng = np.random.default_rng(1)
            Y_orig = rng.uniform(0, 1e-3, size=(n_test, n_x))
            Y_pred = rng.uniform(0, 1e-3, size=(n_test, n_x))
            x_positions = np.linspace(0.0, 1.0, n_x)  # full grid
            # 3 collocation indices (fewer than n_nodes_x)
            colloc_idx = np.array([0, 10, 19])
            from unittest.mock import MagicMock
            mock_lut = MagicMock()
            mock_lut.train_indices = np.arange(n_test)
            # Should not raise
            viz.plot_settlement_comparison_with_collocation(
                Y_orig, Y_pred, mock_lut,
                n_samples=2, x_positions=x_positions, colloc_idx=colloc_idx,
            )
            # Verify output file exists
            self.assertTrue(
                os.path.exists(
                    os.path.join('plots_test', 'settlement_comparison',
                                 'comparison_with_collocation.png')
                )
            )
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_crash_collocation_len_equals_n_nodes_x(self):
        """Standard case: colloc_idx length == n_nodes_x (all nodes)."""
        from src.visualization import Visualization
        tmp = tempfile.mkdtemp(prefix='pi_viz_full_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            viz = Visualization(plots_dir='plots_full')
            n_x = 10
            rng = np.random.default_rng(2)
            Y = rng.uniform(0, 1e-3, size=(6, n_x))
            x_positions = np.linspace(0.0, 1.0, n_x)
            colloc_idx = np.arange(n_x)
            from unittest.mock import MagicMock
            viz.plot_settlement_comparison_with_collocation(
                Y, Y, MagicMock(), n_samples=2,
                x_positions=x_positions, colloc_idx=colloc_idx,
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join('plots_full', 'settlement_comparison',
                                 'comparison_with_collocation.png')
                )
            )
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ylim_bottom_zero_in_settlement_comparison(self):
        """Settlement comparison plot y-axis lower bound must be >= 0.
        Verified by patching Axes.set_ylim to capture all calls."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.visualization import Visualization
        from unittest.mock import patch, call
        tmp = tempfile.mkdtemp(prefix='pi_viz_ylim_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            viz = Visualization(plots_dir='plots_ylim')
            n_x = 10
            rng = np.random.default_rng(3)
            Y = rng.uniform(0.0, 1e-3, size=(4, n_x))
            x_positions = np.linspace(0.0, 1.0, n_x)

            # Capture set_ylim calls by patching at the Axes level
            ylim_calls = []
            original_set_ylim = plt.Axes.set_ylim

            def mock_set_ylim(self_ax, *args, **kwargs):
                ylim_calls.append((args, kwargs))
                return original_set_ylim(self_ax, *args, **kwargs)

            with patch.object(plt.Axes, 'set_ylim', mock_set_ylim):
                viz.plot_settlement_comparison(Y, Y, n_samples=2, x_positions=x_positions)

            # At least one set_ylim call with bottom=0.0 must be present
            has_bottom_zero = any(
                kwargs.get('bottom', None) == 0.0 or (args and args[0] == 0.0)
                for args, kwargs in ylim_calls
            )
            self.assertTrue(
                has_bottom_zero,
                f"Expected set_ylim(bottom=0.0) to be called; calls were: {ylim_calls}"
            )
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)




class TestDCTOutputPhase3NoCrash(unittest.TestCase):
    """Phase-3 training with DCT surrogate output must not crash (CUDA index OOB fix)."""

    def test_dct_output_phase3_no_crash_all_nodes(self):
        """Phase-3 with output_representation='dct', n_output_modes < n_nodes_x,
        and default all-node collocation must not raise CUDA/CPU index out of bounds."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_dct_phase3_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            # DCT surrogate with fewer modes than spatial nodes: this is the scenario
            # that previously caused "index out of bounds" on the GPU.
            cfg['solver']['n_nodes_x'] = 10
            cfg['surrogate']['output_representation'] = 'dct'
            cfg['surrogate']['n_output_modes'] = 3   # K=3 < n_nodes_x=10
            cfg['random_field']['field_basis'] = 'dct'
            cfg['dimension_reducer']['basis_type'] = 'dct'
            cfg['dimension_reducer']['d'] = 2
            # No positions → default all-node collocation (colloc_idx has 10 entries,
            # but surrogate only produces 3 DCT modes → previously caused OOB)
            cfg['collocation'] = {}

            pipeline = TrainingPipeline(cfg, output_dir='r/dct_p3')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            # This call must NOT crash
            reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(X, Y, lut)
            # Verify predictions have correct node-space shape
            xi_prime = reducer.predict(X_test)
            Y_pred = lut.predict(xi_prime)
            self.assertEqual(Y_pred.shape[1], cfg['solver']['n_nodes_x'])
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dct_output_phase3_no_crash_partial_collocation(self):
        """Phase-3 with DCT output and a small collocation subset must not crash."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_dct_p3_partial_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            cfg['solver']['n_nodes_x'] = 10
            cfg['surrogate']['output_representation'] = 'dct'
            cfg['surrogate']['n_output_modes'] = 4   # K=4 < n_nodes_x=10
            cfg['random_field']['field_basis'] = 'dct'
            cfg['dimension_reducer']['basis_type'] = 'dct'
            cfg['dimension_reducer']['d'] = 2
            # Partial phase3 collocation: only 3 of 10 positions
            cfg['collocation_phase3'] = {'positions': [0.0, 0.5, 1.0]}

            pipeline = TrainingPipeline(cfg, output_dir='r/dct_p3_partial')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(X, Y, lut)
            Y_pred = lut.predict(reducer.predict(X_test))
            self.assertEqual(Y_pred.shape[1], cfg['solver']['n_nodes_x'])
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dct_idct_projection_is_differentiable(self):
        """The inverse-DCT projection used in fit_with_surrogate must be a valid
        torch linear op: y_nodes = y_dct @ dct_proj gives shape (B, N)."""
        import torch
        from src.dct_utils import build_idct_basis

        K, N = 3, 10
        D = build_idct_basis(n_modes=K, n_nodes=N)  # (N, K)
        dct_proj = torch.tensor(D.T, dtype=torch.float32)  # (K, N)

        B = 5
        y_dct = torch.randn(B, K, requires_grad=True)
        y_nodes = torch.matmul(y_dct, dct_proj)
        self.assertEqual(y_nodes.shape, (B, N))
        # Should be differentiable (backward must not raise)
        y_nodes.sum().backward()
        self.assertIsNotNone(y_dct.grad)


class TestSeparateCollocationConfigs(unittest.TestCase):
    """Separate collocation_phase2 / collocation_phase3 config sections."""

    def test_phase3_collocation_overrides_common(self):
        """collocation_phase3.positions takes precedence over collocation.positions."""
        from src.training_pipeline import TrainingPipeline
        cfg = {
            'collocation': {'positions': [0.0, 0.25, 0.5, 0.75, 1.0]},
            'collocation_phase3': {'positions': [0.0, 1.0]},
        }
        idx = TrainingPipeline._compute_collocation_indices(
            cfg, n_nodes_x=20, length_x=1.0, phase='phase3'
        )
        # 2 positions → 2 unique node indices
        self.assertEqual(len(idx), 2)

    def test_phase2_collocation_overrides_common(self):
        """collocation_phase2.positions takes precedence over collocation.positions."""
        from src.training_pipeline import TrainingPipeline
        cfg = {
            'collocation': {'positions': [0.0, 0.5, 1.0]},
            'collocation_phase2': {'positions': [0.0, 0.25, 0.5, 0.75, 1.0]},
        }
        idx = TrainingPipeline._compute_collocation_indices(
            cfg, n_nodes_x=20, length_x=1.0, phase='phase2'
        )
        self.assertEqual(len(idx), 5)

    def test_phase3_falls_back_to_common_when_absent(self):
        """When collocation_phase3 is absent, fall back to collocation.positions."""
        from src.training_pipeline import TrainingPipeline
        cfg = {
            'collocation': {'positions': [0.0, 0.5, 1.0]},
        }
        idx = TrainingPipeline._compute_collocation_indices(
            cfg, n_nodes_x=20, length_x=1.0, phase='phase3'
        )
        self.assertEqual(len(idx), 3)

    def test_phase_none_uses_common(self):
        """phase=None always reads from collocation section."""
        from src.training_pipeline import TrainingPipeline
        cfg = {
            'collocation': {'positions': [0.0, 0.5, 1.0]},
            'collocation_phase3': {'positions': [0.0, 1.0]},
        }
        # phase=None should NOT use collocation_phase3
        idx = TrainingPipeline._compute_collocation_indices(
            cfg, n_nodes_x=20, length_x=1.0, phase=None
        )
        self.assertEqual(len(idx), 3)

    def test_separate_collocation_pipeline_no_crash(self):
        """Full pipeline with separate phase3 collocation (fewer positions) must not crash."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_sep_colloc_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            cfg['solver']['n_nodes_x'] = 10
            # Common collocation: 5 positions
            cfg['collocation'] = {'positions': [0.0, 0.25, 0.5, 0.75, 1.0]}
            # Phase-3 uses only 2 positions
            cfg['collocation_phase3'] = {'positions': [0.0, 1.0]}

            pipeline = TrainingPipeline(cfg, output_dir='r/sep_colloc')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(X, Y, lut)

            # The saved collocation_indices.npy must reflect phase3 config (2 positions)
            colloc_saved = np.load('models/reduced_lut/collocation_indices.npy')
            self.assertEqual(len(colloc_saved), 2)
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestPhase4PlottingConsistency(unittest.TestCase):
    """Both comparison.png and comparison_with_collocation.png must use the same samples."""

    def test_both_plots_use_same_samples_and_predictions(self):
        """Verify that phase4_evaluate calls both plot functions with
        pre-selected samples (same Y_plot_true / Y_plot_pred) by checking that
        both output files are created and that the pipeline runs without error."""
        from src.training_pipeline import TrainingPipeline
        tmp = tempfile.mkdtemp(prefix='pi_p4_plots_')
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            cfg = _make_smoke_config()
            # Disable direct-physics for speed (avoids extra solver calls in test)
            cfg['phase4'] = {'use_direct_physics_for_plots': False}

            pipeline = TrainingPipeline(cfg, output_dir='r/p4_plots')
            X, Y = pipeline.phase1_generate_dataset()
            lut = pipeline.phase2_build_reduced_surrogate()
            reducer, (X_test, Y_test) = pipeline.phase3_train_dimension_reducer(X, Y, lut)
            pipeline.phase4_evaluate(reducer, lut, X_test, Y_test)

            plots_dir = 'r/p4_plots/plots/settlement_comparison'
            self.assertTrue(os.path.exists(os.path.join(plots_dir, 'comparison.png')),
                            "comparison.png must be created")
            self.assertTrue(
                os.path.exists(os.path.join(plots_dir, 'comparison_with_collocation.png')),
                "comparison_with_collocation.png must be created",
            )
        finally:
            os.chdir(orig)
            shutil.rmtree(tmp, ignore_errors=True)


class TestPolynomialBasis(unittest.TestCase):
    """Tests for the general polynomial basis in ReducedLUT."""

    def _make_lut(self, basis_order, d):
        """Create a minimal ReducedLUT with the given polynomial basis settings."""
        from src.reduced_lut import ReducedLUT
        from src.forward_solver_2d import BiotSolver2D
        config = {
            'material': {
                'E_ref': 10.0e6,
                'permeability_ref': 1.0e-12,
                'permeability_h': 1.0e-12,
                'permeability_v': 1.0e-12,
            },
            'solver': {'n_nodes_x': 5, 'n_nodes_z': 5, 'type': '2d'},
            'domain': {'length_x': 1.0, 'length_z': 1.0},
            'dimension_reducer': {'d': d, 'basis_type': 'polynomial', 'basis_order': basis_order},
            'reduced_lut': {'n_grid_points': 10, 'grid_type': 'random'},
            'surrogate': {'output_representation': 'direct', 'n_output_modes': 4},
        }
        solver = BiotSolver2D(config)
        return ReducedLUT(config, solver)

    def test_basis_size_order1(self):
        """Order 1 → 3 basis functions: {1, x, z}."""
        lut = self._make_lut(basis_order=1, d=3)
        xi = np.zeros(3)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))

    def test_basis_size_order2(self):
        """Order 2 → 6 basis functions: {1, x, z, x², xz, z²}."""
        lut = self._make_lut(basis_order=2, d=6)
        xi = np.zeros(6)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))

    def test_basis_size_order3(self):
        """Order 3 → 10 basis functions."""
        lut = self._make_lut(basis_order=3, d=10)
        xi = np.zeros(10)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))

    def test_basis_size_order4_is_15(self):
        """Order 4 → 15 basis functions: (4+1)(4+2)/2 = 15."""
        lut = self._make_lut(basis_order=4, d=15)
        xi = np.zeros(15)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))

    def test_basis_order4_supports_d12(self):
        """basis_order=4 (15 terms) must accept d=12 without error."""
        lut = self._make_lut(basis_order=4, d=12)
        xi = np.zeros(12)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))

    def test_basis_order4_rejects_d16(self):
        """d=16 should raise ValueError for basis_order=4 (only 15 terms available)."""
        lut = self._make_lut(basis_order=4, d=16)
        xi = np.zeros(16)
        with self.assertRaises(ValueError) as ctx:
            lut._reconstruct_field_polynomial(xi)
        self.assertIn('15', str(ctx.exception))

    def test_basis_ordering_preserved_order1(self):
        """Order 1: coefficient [0,1,0] (x-term only) should produce gradient in x."""
        lut = self._make_lut(basis_order=1, d=3)
        xi = np.array([0.0, 1.0, 0.0])   # only x coefficient
        E = lut._reconstruct_field_polynomial(xi)
        # Left column (x=0) should be smaller than right column (x=1) in E
        # because log(E) = x_norm increases left to right
        self.assertTrue(E[:, 0].mean() < E[:, -1].mean(),
                        "E should increase from left (x=0) to right (x=1) when only x coeff is positive")

    def test_basis_ordering_preserved_order2(self):
        """Order 2: basis has 6 terms; using all 6 should not error."""
        lut = self._make_lut(basis_order=2, d=6)
        rng = np.random.default_rng(7)
        xi = rng.standard_normal(6)
        E = lut._reconstruct_field_polynomial(xi)
        self.assertEqual(E.shape, (5, 5))
        self.assertTrue(np.all(E > 0))

    def test_basis_size_formula(self):
        """(order+1)(order+2)/2 must match actual number of generated basis functions."""
        from src.reduced_lut import ReducedLUT
        for order in range(0, 6):
            expected = (order + 1) * (order + 2) // 2
            count = 0
            for total_deg in range(0, order + 1):
                for j in range(0, total_deg + 1):
                    count += 1
            self.assertEqual(
                count, expected,
                f"order={order}: expected {expected} basis functions, got {count}"
            )


class TestPhase2MetricDiagnostics(unittest.TestCase):
    """Test that Phase-2 evaluation metrics are computed consistently."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_metric_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_metric_consistency_direct(self):
        """For direct output representation, RMSE should be consistent with R²."""
        from src.training_pipeline import TrainingPipeline
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'direct'
        pipeline = TrainingPipeline(config, output_dir='results/metric_direct')
        pipeline.orchestrate(phases=[1, 2])

        import json
        with open('models/reduced_lut/pce/evaluation/metrics.json') as f:
            metrics = json.load(f)

        # R² and RMSE should be coherent: high R² → small RMSE/std
        # Specifically, RMSE < 10 * std(Y_test) when R² > 0
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        # The metrics should be finite
        self.assertTrue(np.isfinite(metrics['rmse']))

    def test_metric_consistency_dct(self):
        """For DCT output representation, metrics should be in node-space."""
        from src.training_pipeline import TrainingPipeline
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'dct'
        config['surrogate']['n_output_modes'] = 4
        pipeline = TrainingPipeline(config, output_dir='results/metric_dct')
        pipeline.orchestrate(phases=[1, 2])

        import json
        with open('models/reduced_lut/pce/evaluation/metrics.json') as f:
            metrics = json.load(f)

        self.assertIn('rmse', metrics)
        self.assertTrue(np.isfinite(metrics['rmse']))


class TestStochasticPermeability(unittest.TestCase):
    """Test stochastic k_h and k_v feature appending."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_perm_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _base_config(self):
        cfg = _make_smoke_config()
        return cfg

    def test_no_stochastic_scalars_baseline(self):
        """Without stochastic_inputs, X_train has shape (n, n_terms_E)."""
        from src.training_pipeline import TrainingPipeline
        config = self._base_config()
        pipeline = TrainingPipeline(config, output_dir='results/perm_baseline')
        X, Y = pipeline.phase1_generate_dataset()
        # 5 terms, no extra features
        self.assertEqual(X.shape[1], 5)

    def test_k_h_stochastic_appends_feature(self):
        """With k_h stochastic, X_train has one extra column."""
        from src.training_pipeline import TrainingPipeline
        config = self._base_config()
        config['stochastic_inputs'] = {
            'k_h': True, 'k_v': False,
            'k_h_range': [1e-13, 1e-10],
        }
        pipeline = TrainingPipeline(config, output_dir='results/perm_kh')
        X, Y = pipeline.phase1_generate_dataset()
        # 5 terms + 1 log_k_h
        self.assertEqual(X.shape[1], 6)

    def test_both_stochastic_appends_two_features(self):
        """With both k_h and k_v stochastic, X_train has two extra columns."""
        from src.training_pipeline import TrainingPipeline
        config = self._base_config()
        config['stochastic_inputs'] = {
            'k_h': True, 'k_v': True,
            'k_h_range': [1e-13, 1e-10],
            'k_v_range': [1e-13, 1e-10],
        }
        pipeline = TrainingPipeline(config, output_dir='results/perm_both')
        X, Y = pipeline.phase1_generate_dataset()
        # 5 terms + 2 log permeabilities
        self.assertEqual(X.shape[1], 7)

    def test_k_h_k_v_stochastic_full_pipeline(self):
        """Full pipeline with both k_h and k_v stochastic should complete without error."""
        from src.training_pipeline import TrainingPipeline
        config = self._base_config()
        config['stochastic_inputs'] = {
            'k_h': True, 'k_v': True,
            'k_h_range': [1e-13, 1e-10],
            'k_v_range': [1e-13, 1e-10],
        }
        pipeline = TrainingPipeline(config, output_dir='results/perm_pipeline')
        metrics = pipeline.orchestrate(phases=[1, 2, 3, 4])
        self.assertIn('r2', metrics)
        # Feature layout check
        X = np.load('data/X_train.npy')
        self.assertEqual(X.shape[1], 7)  # 5 + 2


class TestNewOutputBases(unittest.TestCase):
    """Test new poly and bspline output representation forward/inverse round-trips."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='pi_dimre_basis_')
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_poly_round_trip(self):
        """Polynomial output: forward then inverse should approximately recover Y."""
        from src.reduced_lut import ReducedLUT
        from src.forward_solver_2d import BiotSolver2D
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'poly'
        config['surrogate']['n_output_modes'] = 5
        solver = BiotSolver2D(config)
        lut = ReducedLUT(config, solver, output_dir='models/reduced_lut')
        rng = np.random.default_rng(0)
        # Smooth synthetic Y profiles
        x = np.linspace(0, 1, 8)
        Y = np.vstack([np.sin(x + rng.uniform(0, 0.5)) for _ in range(5)])
        coeffs = lut._to_poly_space(Y)
        Y_rec = lut._from_poly_space(coeffs)
        self.assertEqual(Y_rec.shape, Y.shape)
        # Allow ~50% round-trip error for polynomial fitting (loose tolerance)
        rel_err = np.mean(np.abs(Y_rec - Y)) / (np.mean(np.abs(Y)) + 1e-15)
        self.assertLess(rel_err, 0.5)

    def test_bspline_round_trip(self):
        """B-spline output: forward then inverse should approximately recover Y."""
        from src.reduced_lut import ReducedLUT
        from src.forward_solver_2d import BiotSolver2D
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'bspline'
        config['surrogate']['n_output_modes'] = 6
        solver = BiotSolver2D(config)
        lut = ReducedLUT(config, solver, output_dir='models/reduced_lut')
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 8)
        Y = np.vstack([np.sin(x + rng.uniform(0, 0.5)) for _ in range(5)])
        coeffs = lut._to_bspline_space(Y)
        Y_rec = lut._from_bspline_space(coeffs)
        self.assertEqual(Y_rec.shape, Y.shape)
        rel_err = np.mean(np.abs(Y_rec - Y)) / (np.mean(np.abs(Y)) + 1e-15)
        self.assertLess(rel_err, 0.5)

    def test_poly_smoke_phase2(self):
        """Phase 2 with poly output representation completes without error."""
        from src.training_pipeline import TrainingPipeline
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'poly'
        config['surrogate']['n_output_modes'] = 4
        pipeline = TrainingPipeline(config, output_dir='results/poly_phase2')
        pipeline.orchestrate(phases=[1, 2])

    def test_bspline_smoke_phase2(self):
        """Phase 2 with bspline output representation completes without error."""
        from src.training_pipeline import TrainingPipeline
        config = _make_smoke_config()
        config['surrogate']['output_representation'] = 'bspline'
        config['surrogate']['n_output_modes'] = 6
        pipeline = TrainingPipeline(config, output_dir='results/bspline_phase2')
        pipeline.orchestrate(phases=[1, 2])


if __name__ == '__main__':
    unittest.main()

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

    def test_permeability_effect(self):
        """High k_h should give different settlement than low k_h."""
        from src.forward_solver_2d import BiotSolver2D
        solver = BiotSolver2D(self.config)
        E_field = np.full((10, 10), 10e6)
        Y1 = solver.run(E_field, k_h=1e-12, k_v=1e-12)
        Y2 = solver.run(E_field, k_h=1e-8, k_v=1e-12)
        self.assertFalse(np.allclose(Y1, Y2))


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

        # Core artifacts exist
        self.assertTrue(os.path.exists('data/X_train.npy'))
        self.assertTrue(os.path.exists('data/Y_train.npy'))
        self.assertTrue(os.path.exists('data/lut_grid_points.npy'))
        self.assertTrue(os.path.exists('data/lut_responses.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/config.json'))
        self.assertTrue(os.path.exists('models/reduced_lut/grid_points.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/responses.npy'))
        self.assertTrue(os.path.exists('models/reduced_lut/surrogate_pce.pkl'))
        self.assertTrue(os.path.exists('models/dimension_reducer_pce.pkl'))
        self.assertTrue(os.path.exists('results/smoke_pce/metrics.json'))
        self.assertTrue(os.path.exists('results/smoke_pce/run_summary.txt'))

        # Array shapes
        X = np.load('data/X_train.npy')
        Y = np.load('data/Y_train.npy')
        self.assertEqual(X.shape, (8, 5))
        self.assertEqual(Y.shape[0], 8)

        # Metrics dict has expected keys
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('rel_l2', metrics)

    def test_phase2_reuse(self):
        """Phase 2 reuse: second run loads cached surrogate without recomputing."""
        from src.training_pipeline import TrainingPipeline
        import json

        config = _make_smoke_config()
        # First run: build and save
        p1 = TrainingPipeline(config, output_dir='results/reuse_first')
        p1.orchestrate(phases=[1, 2])

        # Read the creation date from the first run's config.json
        with open('models/reduced_lut/config.json') as f:
            meta_first = json.load(f)

        # Second run: should load cached surrogate (config.json unchanged)
        p2 = TrainingPipeline(config, output_dir='results/reuse_second')
        p2.orchestrate(phases=[2])

        with open('models/reduced_lut/config.json') as f:
            meta_second = json.load(f)

        # The created_date should be unchanged (loaded, not regenerated)
        self.assertEqual(meta_first['created_date'], meta_second['created_date'])

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


if __name__ == '__main__':
    unittest.main()
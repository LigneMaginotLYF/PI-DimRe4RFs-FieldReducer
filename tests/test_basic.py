import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigManager(unittest.TestCase):
    def test_load_config(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager(config_file='config.yaml')
        self.assertIn('dataset', cm.config)
        self.assertIn('material', cm.config)
        self.assertIn('solver', cm.config)

    def test_deep_merge(self):
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        override = {'a': {'b': 10}}
        result = cm._deep_merge(base, override)
        self.assertEqual(result['a']['b'], 10)
        self.assertEqual(result['a']['c'], 2)
        self.assertEqual(result['d'], 3)


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


if __name__ == '__main__':
    unittest.main()
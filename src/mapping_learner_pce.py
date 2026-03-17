import numpy as np
from itertools import product
import logging

logger = logging.getLogger(__name__)


def hermite_poly(n, x):
    """Probabilist's Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        Hm2 = np.ones_like(x)
        Hm1 = x.copy()
        for k in range(2, n + 1):
            H = x * Hm1 - (k - 1) * Hm2
            Hm2 = Hm1
            Hm1 = H
        return Hm1


class PolynomialChaosExpansion:
    """
    Polynomial Chaos Expansion surrogate using Hermite polynomials.
    Handles multi-input, multi-output case.
    """

    def __init__(self, degree=3, n_inputs=3, n_outputs=20):
        self.degree = degree
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.multi_indices = None
        self.coefficients = None
        self._build_multi_indices()

    def _build_multi_indices(self):
        """Build all multi-indices with total degree <= self.degree."""
        indices = []
        for alpha in product(range(self.degree + 1), repeat=self.n_inputs):
            if sum(alpha) <= self.degree:
                indices.append(alpha)
        self.multi_indices = np.array(indices)
        logger.info(f"PCE: {len(self.multi_indices)} basis functions (degree={self.degree}, inputs={self.n_inputs})")

    def _eval_basis(self, X):
        """
        Evaluate all basis functions at X.
        Args:
            X: shape (n_samples, n_inputs)
        Returns:
            Phi: shape (n_samples, n_basis)
        """
        n_samples = X.shape[0]
        n_basis = len(self.multi_indices)
        Phi = np.ones((n_samples, n_basis))
        for b_idx, alpha in enumerate(self.multi_indices):
            for d_idx in range(self.n_inputs):
                if alpha[d_idx] > 0:
                    Phi[:, b_idx] *= hermite_poly(alpha[d_idx], X[:, d_idx])
        return Phi

    def fit(self, X_train, Y_train):
        """
        Fit PCE coefficients using ordinary least squares.
        Args:
            X_train: shape (n_samples, n_inputs)
            Y_train: shape (n_samples, n_outputs)
        """
        Phi = self._eval_basis(X_train)
        self.coefficients, _, _, _ = np.linalg.lstsq(Phi, Y_train, rcond=None)
        logger.info(f"PCE fit: {X_train.shape[0]} samples, condition number ~{np.linalg.cond(Phi):.2e}")
        return self

    def fit_with_surrogate(self, X_train, Y_train, surrogate, colloc_idx=None,
                           output_representation='direct', n_output_modes=None,
                           n_nodes_x=None):
        """
        Fit PCE as dimension reducer M: xi_E -> xi' using surrogate-based loss.

        Loss is always computed in *node space* so that collocation indices
        (which are node indices 0..n_x-1) remain valid regardless of the
        surrogate's internal output representation.

        When ``output_representation='dct'``, the surrogate returns K DCT
        coefficients.  These are mapped to full node-space predictions via an
        inverse-DCT projection before the collocation mask is applied.

        Uses a two-step approach: first fit PCE to map xi_E -> xi' by minimizing
        ||S(M(xi_E))[:, colloc_idx] - Y_train[:, colloc_idx]||, approximated by
        projecting onto surrogate predictions.

        Args:
            X_train: shape (n_samples, n_inputs) - KL coefficients xi_E
            Y_train: shape (n_samples, n_x) - reference settlement profiles
            surrogate: fitted surrogate S: xi'(d) -> (K,) in DCT mode or (n_x,)
            colloc_idx: 1-D integer array of node indices (0..n_x-1) to include
                in the loss.  If None, all nodes are used.
            output_representation: 'direct' | 'dct'.
            n_output_modes: K, number of DCT modes (required for DCT mode).
            n_nodes_x: N, number of spatial nodes (required for DCT mode).
        """
        from scipy.optimize import minimize

        # Pre-compute numpy inverse-DCT projection matrix for DCT mode.
        # idct_basis has shape (N, K); the node-space prediction for a
        # coefficient vector c (length K) is: y_nodes = idct_basis @ c
        idct_basis = None
        if (output_representation == 'dct'
                and n_output_modes is not None
                and n_nodes_x is not None):
            from src.dct_utils import build_idct_basis
            idct_basis = build_idct_basis(n_modes=n_output_modes, n_nodes=n_nodes_x)
            if colloc_idx is not None and len(colloc_idx) > int(n_output_modes):
                logger.warning(
                    f"Phase-3 PCE: collocation covers {len(colloc_idx)} nodes "
                    f"but surrogate has only {n_output_modes} DCT output modes. "
                    "Loss is computed in node space via inverse-DCT projection."
                )

        n_samples = X_train.shape[0]
        xi_prime_targets = np.zeros((n_samples, self.n_outputs))

        for i in range(n_samples):
            y_ref = Y_train[i]

            def obj(xi_p):
                y_raw = surrogate.predict(xi_p.reshape(1, -1))[0]
                # Convert to node space when surrogate outputs DCT coefficients
                if idct_basis is not None:
                    y_pred = idct_basis @ y_raw
                else:
                    y_pred = y_raw
                if colloc_idx is not None:
                    return np.sum((y_pred[colloc_idx] - y_ref[colloc_idx]) ** 2)
                return np.sum((y_pred - y_ref) ** 2)

            def grad(xi_p):
                # Finite-difference gradient of obj.  Because obj already
                # filters outputs through colloc_idx, this gradient is
                # correctly restricted to the collocation subset.
                eps = 1e-5
                g = np.zeros_like(xi_p)
                f0 = obj(xi_p)
                for k in range(len(xi_p)):
                    xi_p[k] += eps
                    g[k] = (obj(xi_p) - f0) / eps
                    xi_p[k] -= eps
                return g

            x0 = np.zeros(self.n_outputs)
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10})
            xi_prime_targets[i] = res.x

        self.fit(X_train, xi_prime_targets)
        return self

    def predict(self, X):
        """
        Predict for new inputs.
        Args:
            X: shape (n_samples, n_inputs) or (n_inputs,)
        Returns:
            Y_pred: shape (n_samples, n_outputs) or (n_outputs,)
        """
        scalar = X.ndim == 1
        if scalar:
            X = X.reshape(1, -1)
        Phi = self._eval_basis(X)
        Y_pred = Phi @ self.coefficients
        return Y_pred[0] if scalar else Y_pred

import numpy as np
from scipy.special import kv as bessel_kv, gamma
from scipy.spatial.distance import cdist


class MaternKernel:
    """Matern covariance kernel."""

    def __init__(self, nu, length_scale, variance=1.0):
        self.nu = nu
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        D = cdist(X, Y, metric='euclidean')
        return self._matern(D)

    def _matern(self, D):
        nu = self.nu
        ls = self.length_scale
        sigma2 = self.variance
        D_safe = np.where(D == 0, 1e-10, D)
        z = np.sqrt(2 * nu) * D_safe / ls
        K = sigma2 * (2 ** (1 - nu) / gamma(nu)) * (z ** nu) * bessel_kv(nu, z)
        K = np.where(D == 0, sigma2, K)
        return K


class KLExpansionField:
    """Generate 2D random field using KL expansion with Matern covariance."""

    def __init__(self, n_nodes_x, n_nodes_z, length_x=1.0, length_z=1.0):
        self.n_nodes_x = n_nodes_x
        self.n_nodes_z = n_nodes_z
        self.length_x = length_x
        self.length_z = length_z
        x = np.linspace(0, length_x, n_nodes_x)
        z = np.linspace(0, length_z, n_nodes_z)
        X, Z = np.meshgrid(x, z)
        self.grid_points = np.column_stack([X.ravel(), Z.ravel()])
        self.n_pts = n_nodes_x * n_nodes_z

    def compute_kl_basis(self, nu, length_scale, n_terms):
        """Compute KL eigenpairs for given Matern parameters."""
        kernel = MaternKernel(nu=nu, length_scale=length_scale)
        C = kernel(self.grid_points)
        C += 1e-8 * np.eye(self.n_pts)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        lambdas = eigenvalues[:n_terms]
        phis = eigenvectors[:, :n_terms]
        return lambdas, phis

    def generate_field(self, xi, nu, length_scale, n_terms=5, E_ref=10.0e6):
        """
        Generate Young's modulus field.
        E(x,z) = E_ref * exp(sum_k sqrt(lambda_k) * phi_k(x,z) * xi_k)

        Args:
            xi: KL coefficients, shape (n_terms,)
            nu: Matern smoothness
            length_scale: Matern length scale
            n_terms: number of KL terms
            E_ref: reference Young's modulus [Pa]
        Returns:
            E_field: shape (n_nodes_z, n_nodes_x)
        """
        lambdas, phis = self.compute_kl_basis(nu, length_scale, n_terms)
        log_E = phis @ (np.sqrt(lambdas) * xi)
        E = E_ref * np.exp(log_E)
        return E.reshape(self.n_nodes_z, self.n_nodes_x)

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

    def generate_field(self, xi, nu, length_scale, n_terms=5, E_ref=10.0e6,
                       logE_std=1.0):
        """
        Generate Young's modulus field.
        E(x,z) = E_ref * exp(logE_std * sum_k sqrt(lambda_k) * phi_k(x,z) * xi_k)

        Args:
            xi: KL coefficients, shape (n_terms,)
            nu: Matern smoothness
            length_scale: Matern length scale
            n_terms: number of KL terms
            E_ref: reference Young's modulus [Pa]
            logE_std: global multiplier on logE amplitude (default 1.0)
        Returns:
            E_field: shape (n_nodes_z, n_nodes_x)
        """
        lambdas, phis = self.compute_kl_basis(nu, length_scale, n_terms)
        log_E = logE_std * (phis @ (np.sqrt(lambdas) * xi))
        E = E_ref * np.exp(np.clip(log_E, -10, 10))
        return E.reshape(self.n_nodes_z, self.n_nodes_x)


class DCTField:
    """
    Generate 2D random field using a fixed 2D DCT-II basis on log(E).

    The basis functions psi_k(x,z) are deterministic 2D DCT-II modes, ordered
    by 2D spatial frequency magnitude (lowest-frequency / smoothest modes first).
    The basis is independent of Matérn parameters, so two samples with the same
    coefficient vector produce the same log-field regardless of (nu, length_scale).

    Matérn-like spatial structure is obtained by shaping the variance of each
    coefficient a_k according to an approximate 2D Matérn spectral density:

        var(a_k) ∝ (2·ν/ℓ² + ‖ω_k‖²)^{-(ν+1)}

    where ω_k = (m·π/L_x, n·π/L_z) is the angular frequency of mode k.
    This is an approximation to the exact Matérn spectral density in 2D
    (which has exponent -(ν+1) for 2D, i.e. spectral density ∝ (α²+ω²)^{-(ν+d/2)}
    with d=2).  The formula is documented here so users can audit or replace it.

    The field is:
        logE(x,z) = logE_std · sum_k a_k · psi_k(x,z)
        E(x,z)    = E_ref · exp(clip(logE, -10, 10))

    where a_k ~ N(0, sigma_k²) are drawn independently per sample.

    Usage (Phase 1):
        field = DCTField(n_nodes_x, n_nodes_z, length_x, length_z)
        E     = field.generate_field(rng, nu, length_scale,
                                     n_terms=n_terms, E_ref=E_ref, logE_std=1.0)

    Usage (Phase 2 reconstruction from fixed coefficients):
        E = field.reconstruct_from_coefficients(xi, E_ref=E_ref, logE_std=1.0)
    """

    def __init__(self, n_nodes_x, n_nodes_z, length_x=1.0, length_z=1.0):
        self.n_nodes_x = n_nodes_x
        self.n_nodes_z = n_nodes_z
        self.length_x = length_x
        self.length_z = length_z
        self._basis_cache = {}  # cache keyed by n_terms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_dct_basis(self, n_terms):
        """
        Build and cache the (n_pts, n_terms) DCT-II basis matrix.

        Modes are ordered by 2D angular frequency magnitude ‖ω_k‖ (ascending),
        so that the first n_terms modes capture the smoothest spatial variation.
        The basis matrix is L2-normalised so that each column has unit norm.

        Returns
        -------
        Psi : ndarray, shape (n_pts, n_terms)
            Basis matrix; columns are fixed DCT modes.
        mode_freqs : ndarray, shape (n_terms,)
            Angular frequency magnitude for each selected mode.
        """
        if n_terms in self._basis_cache:
            return self._basis_cache[n_terms]

        nx = self.n_nodes_x
        nz = self.n_nodes_z
        Lx = self.length_x
        Lz = self.length_z
        n_pts = nx * nz

        # Build all 2D DCT-II basis functions for modes (m, n)
        # where m in 0..nx-1, n in 0..nz-1
        # psi_{m,n}(i,j) = c_m * c_n * cos(pi*m*(i+0.5)/nx) * cos(pi*n*(j+0.5)/nz)
        # with c_0 = 1/sqrt(N), c_k = sqrt(2/N)
        ix = np.arange(nx)
        jz = np.arange(nz)
        m_idx = np.arange(nx)
        n_idx = np.arange(nz)

        cos_x = np.cos(np.pi * m_idx[:, None] * (ix[None, :] + 0.5) / nx)  # (nx, nx)
        cos_z = np.cos(np.pi * n_idx[:, None] * (jz[None, :] + 0.5) / nz)  # (nz, nz)

        cx = np.where(m_idx == 0, 1.0 / np.sqrt(nx), np.sqrt(2.0 / nx))
        cz = np.where(n_idx == 0, 1.0 / np.sqrt(nz), np.sqrt(2.0 / nz))

        # Normalised 1-D DCT vectors: shape (nx, nx) and (nz, nz)
        phi_x = cx[:, None] * cos_x   # (nx, nx): row m, col i
        phi_z = cz[:, None] * cos_z   # (nz, nz): row n, col j

        # 2D angular frequency magnitudes for all (m,n) pairs
        omega_x = m_idx * np.pi / Lx  # (nx,)
        omega_z = n_idx * np.pi / Lz  # (nz,)
        omega_mn = np.sqrt(omega_x[:, None] ** 2 + omega_z[None, :] ** 2)  # (nx, nz)

        # Flatten mode index ordering by frequency magnitude
        flat_omega = omega_mn.ravel()          # (nx*nz,)
        order = np.argsort(flat_omega)         # ascending freq order
        selected = order[:n_terms]            # first n_terms modes

        # Build basis matrix Psi of shape (n_pts, n_terms)
        # Grid convention: row-major z-major (Z varies with rows in meshgrid)
        # The field array has shape (nz, nx) → flattened order is z-major
        Psi = np.zeros((n_pts, n_terms))
        for k, flat_k in enumerate(selected):
            m = flat_k // nz
            n = flat_k % nz
            psi_2d = np.outer(phi_z[n, :], phi_x[m, :])  # (nz, nx)
            Psi[:, k] = psi_2d.ravel()

        self._basis_cache[n_terms] = (Psi, flat_omega[selected])
        return Psi, flat_omega[selected]

    def matern_spectral_variance(self, nu, length_scale, mode_freqs):
        """
        Compute approximate Matérn spectral variance for each DCT mode.

        Uses the 2D Matérn spectral density (up to a constant):
            S(ω) ∝ (2ν/ℓ² + ω²)^{-(ν+1)}

        The result is normalised so that its sum equals 1, giving relative
        per-mode variances.  Multiply by logE_std² to get absolute variances.

        Parameters
        ----------
        nu : float
            Matérn smoothness parameter.
        length_scale : float
            Matérn length scale ℓ.
        mode_freqs : ndarray, shape (n_terms,)
            Angular frequency magnitude ‖ω_k‖ for each mode.

        Returns
        -------
        sigma_k : ndarray, shape (n_terms,)
            Standard deviation for each coefficient a_k.
        """
        alpha2 = 2.0 * nu / (length_scale ** 2)
        S = (alpha2 + mode_freqs ** 2) ** (-(nu + 1.0))
        S = np.maximum(S, 1e-30)
        # Normalise so that sum(sigma_k^2) = 1 (unit total variance before logE_std)
        sigma_k = np.sqrt(S / S.sum())
        return sigma_k

    def generate_field(self, rng, nu, length_scale, n_terms=5, E_ref=10.0e6,
                       logE_std=1.0):
        """
        Draw a random DCT-basis log-E field and return the E field.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator (for reproducibility).
        nu : float
            Matérn smoothness (shapes coefficient variance; does NOT change basis).
        length_scale : float
            Matérn length scale (shapes coefficient variance; does NOT change basis).
        n_terms : int
            Number of DCT modes to use.
        E_ref : float
            Reference Young's modulus [Pa].
        logE_std : float
            Global amplitude multiplier on logE (default 1.0; decrease to smooth GT).

        Returns
        -------
        xi : ndarray, shape (n_terms,)
            Sampled DCT coefficients (stored as the "input" features X in Phase 1).
        E_field : ndarray, shape (n_nodes_z, n_nodes_x)
            Young's modulus field.
        """
        Psi, mode_freqs = self.compute_dct_basis(n_terms)
        sigma_k = self.matern_spectral_variance(nu, length_scale, mode_freqs)
        xi = rng.standard_normal(n_terms) * sigma_k
        log_E = logE_std * (Psi @ xi)
        E = E_ref * np.exp(np.clip(log_E, -10, 10))
        return xi, E.reshape(self.n_nodes_z, self.n_nodes_x)

    def reconstruct_from_coefficients(self, xi, E_ref=10.0e6, logE_std=1.0):
        """
        Reconstruct E field from a fixed coefficient vector xi (no randomness).

        Parameters
        ----------
        xi : ndarray, shape (n_terms,)
            DCT coefficients.
        E_ref : float
        logE_std : float

        Returns
        -------
        E_field : ndarray, shape (n_nodes_z, n_nodes_x)
        """
        n_terms = len(xi)
        Psi, _ = self.compute_dct_basis(n_terms)
        log_E = logE_std * (Psi @ xi)
        E = E_ref * np.exp(np.clip(log_E, -10, 10))
        return E.reshape(self.n_nodes_z, self.n_nodes_x)

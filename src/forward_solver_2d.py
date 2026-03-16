import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class BiotSolver2D:
    """
    2D Biot consolidation solver using finite differences.

    Supports both *steady-state* and *transient* response modes controlled by
    ``solver.response_mode`` in the configuration dict.

    **Steady-state** flow equation::

        k_h * d²p/dx² + k_v * d²p/dz² = 0

    **Transient** flow equation (Terzaghi-Biot diffusion)::

        ∂p/∂t = k_h/γ_w * ∂²p/∂x² + k_v/γ_w * ∂²p/∂z²

    Boundary conditions (both modes):
      - Bottom (z=0): p = p_bottom (uniform fixed pressure)
      - Top (z=L_z): p = 0 (fully permeable, all nodes drain)
      - Left/Right sides: dp/dn = 0 (no-flow)

    Settlement: u_z(x) = sum_j [ (q - b*p[j,i]) / M_v[j,i] * dz ]
    """

    def __init__(self, config):
        mat = config.get('material', {})
        dom = config.get('domain', {})
        sol = config.get('solver', {})

        self.n_x = sol.get('n_nodes_x', 20)
        self.n_z = sol.get('n_nodes_z', 20)
        self.L_x = dom.get('length_x', 1.0)
        self.L_z = dom.get('length_z', 1.0)
        self.nu = mat.get('poisson_ratio', 0.3)
        self.b = mat.get('biot_coefficient', 0.8)
        self.q = mat.get('applied_load', 1.0e6)
        self.p_bottom = mat.get('pore_pressure_bottom', 1.0e5)

        self.response_mode = sol.get('response_mode', 'steady_state')
        # Non-dimensional final time (T = c_v · t / L_z²; 1.0 ≈ 95% consolidation)
        self.T_final = float(sol.get('t_final', 1.0))
        # Maximum explicit time steps (configurable via solver.max_time_steps).
        self.max_time_steps = int(sol.get('max_time_steps', 2000))

        self.dx = self.L_x / (self.n_x - 1)
        self.dz = self.L_z / (self.n_z - 1)

    def _idx(self, i, j):
        """Linear index from (i=x-index, j=z-index)."""
        return j * self.n_x + i

    def _build_flow_matrix(self, k_h, k_v):
        """Build sparse FD matrix for the Darcy flow equation."""
        N = self.n_x * self.n_z
        rows, cols, vals = [], [], []

        dx2 = self.dx ** 2
        dz2 = self.dz ** 2

        for j in range(self.n_z):
            for i in range(self.n_x):
                row = self._idx(i, j)

                if j == 0:
                    rows.append(row); cols.append(row); vals.append(1.0)
                    continue

                if j == self.n_z - 1:
                    # Top surface: fully permeable (Dirichlet p=0 for all x nodes)
                    rows.append(row); cols.append(row); vals.append(1.0)
                    continue

                diag = 0.0

                if i == 0:
                    diag -= 2 * k_h / dx2
                    rows.append(row); cols.append(self._idx(i + 1, j)); vals.append(2 * k_h / dx2)
                elif i == self.n_x - 1:
                    diag -= 2 * k_h / dx2
                    rows.append(row); cols.append(self._idx(i - 1, j)); vals.append(2 * k_h / dx2)
                else:
                    diag -= 2 * k_h / dx2
                    rows.append(row); cols.append(self._idx(i + 1, j)); vals.append(k_h / dx2)
                    rows.append(row); cols.append(self._idx(i - 1, j)); vals.append(k_h / dx2)

                diag -= 2 * k_v / dz2
                rows.append(row); cols.append(self._idx(i, j + 1)); vals.append(k_v / dz2)
                rows.append(row); cols.append(self._idx(i, j - 1)); vals.append(k_v / dz2)

                rows.append(row); cols.append(row); vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return A

    def _build_rhs(self):
        """Build RHS for the flow system."""
        N = self.n_x * self.n_z
        b = np.zeros(N)
        for j in range(self.n_z):
            for i in range(self.n_x):
                row = self._idx(i, j)
                if j == 0:
                    b[row] = self.p_bottom
                elif j == self.n_z - 1:
                    b[row] = 0.0
        return b

    def solve_flow(self, k_h, k_v):
        """Solve steady-state Darcy flow, return pressure field shape (n_z, n_x)."""
        A = self._build_flow_matrix(k_h, k_v)
        rhs = self._build_rhs()
        p_flat = spla.spsolve(A, rhs)
        return p_flat.reshape(self.n_z, self.n_x)

    def compute_settlement(self, E_field, p_field):
        """
        Compute settlement profile.

        u_z(x_i) = sum_j (q - b*p[j,i]) / M_v[j,i] * dz

        Args:
            E_field: shape (n_z, n_x)
            p_field: shape (n_z, n_x)
        Returns:
            settlement: shape (n_x,)
        """
        nu = self.nu
        b = self.b
        q = self.q
        dz = self.dz
        M_v = E_field * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        sigma_prime_z = q - b * p_field
        eps_z = sigma_prime_z / M_v
        settlement = np.sum(eps_z, axis=0) * dz
        return settlement

    def run(self, E_field, k_h, k_v):
        """
        Run complete 2D Biot solver (steady-state or transient).

        Args:
            E_field: Young's modulus field, shape (n_z, n_x) [Pa]
            k_h: horizontal permeability [m²]
            k_v: vertical permeability [m²]
        Returns:
            settlement: surface settlement profile, shape (n_x,) [m]
        """
        if self.response_mode == 'steady_state':
            p_field = self.solve_flow(k_h, k_v)
        elif self.response_mode == 'transient':
            p_field = self._solve_transient(k_h, k_v, E_field)
        else:
            raise ValueError(
                f"Unknown response_mode {self.response_mode!r}. "
                "Use 'steady_state' or 'transient'."
            )
        return self.compute_settlement(E_field, p_field)

    def _solve_transient(self, k_h, k_v, E_field):
        """
        Forward-Euler time-stepping for 2-D Biot diffusion.

        Returns the pressure field at t = T_final.
        """
        M_v = E_field * (1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        gamma_w = 1.0e4  # N/m³ (unit weight of water)
        c_v = float(k_v * np.mean(M_v)) / gamma_w  # representative diffusivity

        c_v_safe = max(c_v, 1.0e-30)
        t_final = self.T_final * self.L_z ** 2 / c_v_safe

        dt_stable = 0.25 * min(self.dx, self.dz) ** 2 / c_v_safe
        n_steps = max(50, int(np.ceil(t_final / dt_stable)))
        _max_steps = int(self.max_time_steps)
        if n_steps > _max_steps:
            import warnings
            warnings.warn(
                f"BiotSolver2D transient: required {n_steps} time steps for "
                f"stability, but capped at {_max_steps} (solver.max_time_steps). "
                "Physical accuracy may be reduced. Increase max_time_steps or "
                "use a coarser grid to improve fidelity.",
                UserWarning, stacklevel=3,
            )
        n_steps = min(n_steps, _max_steps)
        dt = t_final / n_steps

        rx = c_v_safe * dt / self.dx ** 2
        rz = c_v_safe * dt / self.dz ** 2
        # Clamp to enforce von-Neumann stability
        scale = max(1.0, (rx + rz) / 0.49)
        rx, rz = rx / scale, rz / scale

        # Initial excess pore pressure = b * q (undrained response at t=0)
        p = np.full((self.n_z, self.n_x), self.b * self.q)
        p[0, :] = self.p_bottom   # bottom BC
        p[-1, :] = 0.0            # top BC

        for _ in range(n_steps):
            p_new = p.copy()
            p_new[1:-1, 1:-1] = (
                p[1:-1, 1:-1]
                + rx * (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2])
                + rz * (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1])
            )
            # Left/right: no-flow (Neumann) → copy interior
            p_new[1:-1, 0] = p_new[1:-1, 1]
            p_new[1:-1, -1] = p_new[1:-1, -2]
            # Top/bottom BCs
            p_new[0, :] = self.p_bottom
            p_new[-1, :] = 0.0
            p = p_new

        return p


class BiotsConsolidationSolver2D(BiotSolver2D):
    """Backward-compatible alias for BiotSolver2D."""

    def __init__(self, parameters):
        self.parameters = parameters
        config = {
            'material': parameters,
            'domain': parameters.get('domain', {'length_x': 1.0, 'length_z': 1.0}),
            'solver': parameters.get('solver', {'n_nodes_x': 20, 'n_nodes_z': 20}),
        }
        super().__init__(config)

    def solve(self):
        raise NotImplementedError("Use run(E_field, k_h, k_v) instead")

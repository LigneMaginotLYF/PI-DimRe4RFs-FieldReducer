import numpy as np


class BiotSolver1D:
    """
    1D Biot consolidation solver (z-direction only).

    The domain is a single vertical soil column [0, L_z].

    Steady-state flow: d²p/dz² = 0 → p(z) = p_bottom * (1 - z/L_z)
    Boundary conditions:
      - Bottom (z=0): p = p_bottom (fixed pore pressure)
      - Top    (z=L_z): p = 0 (fully permeable / drained)

    Transient flow: dp/dt = c_v(z) * d²p/dz²
    Initial condition: p(z, 0) = b * q  (instantaneous application of surface load)

    Settlement (surface):
      u_z = sum_j  (q - b * p[j]) / M_v[j] * dz

    The method ``run`` returns shape ``(n_x,)`` where ``n_x = solver.n_nodes_x``
    from config.  All elements of the array are equal (1-D problem has no
    x-variation), which keeps the interface identical to ``BiotSolver2D``.
    """

    def __init__(self, config):
        mat = config.get('material', {})
        dom = config.get('domain', {})
        sol = config.get('solver', {})

        self.n_z = sol.get('n_nodes_z', 20)
        # n_x controls the *output* array length for interface compatibility.
        # For a pure 1-D problem set n_nodes_x=1.
        self.n_x = sol.get('n_nodes_x', 1)
        self.L_z = dom.get('length_z', 1.0)

        self.nu = mat.get('poisson_ratio', 0.3)
        self.b = mat.get('biot_coefficient', 0.8)
        self.q = mat.get('applied_load', 1.0e6)
        self.p_bottom = mat.get('pore_pressure_bottom', 1.0e5)

        self.response_mode = sol.get('response_mode', 'steady_state')
        # Non-dimensional final time T = c_v * t / L_z^2.
        # T_final ≈ 1.0 corresponds to ~95 % primary consolidation.
        self.T_final = float(sol.get('t_final', 1.0))
        # Maximum number of explicit time steps (configurable via solver.max_time_steps).
        # Increase for production runs requiring high physical accuracy.
        self.max_time_steps = int(sol.get('max_time_steps', 2000))

        self.dz = self.L_z / (self.n_z - 1)

    # ------------------------------------------------------------------
    def run(self, E_field, k_h, k_v):
        """
        Run 1-D Biot solver.

        Parameters
        ----------
        E_field : array-like
            Young's modulus field.  Accepts shapes ``(n_z,)``, ``(n_z, 1)``,
            ``(1, n_z)``, or any flat array with at least ``n_z`` elements.
        k_h : float
            Horizontal permeability (unused in 1-D; kept for API compatibility).
        k_v : float
            Vertical permeability [m²].

        Returns
        -------
        settlement : np.ndarray, shape ``(n_x,)``
            Surface settlement (all values equal – no x-variation).
        """
        E_1d = np.asarray(E_field, dtype=float).ravel()[:self.n_z]

        if self.response_mode == 'steady_state':
            u_z = self._solve_steady(E_1d)
        elif self.response_mode == 'transient':
            u_z = self._solve_transient(E_1d, float(k_v))
        else:
            raise ValueError(
                f"Unknown response_mode {self.response_mode!r}. "
                "Use 'steady_state' or 'transient'."
            )

        return np.full(self.n_x, u_z)

    # ------------------------------------------------------------------
    def _m_v(self, E_1d):
        """Constrained modulus M_v = E(1-ν)/((1+ν)(1-2ν))."""
        return E_1d * (1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def _solve_steady(self, E_1d):
        """Closed-form steady-state settlement."""
        z = np.linspace(0.0, self.L_z, self.n_z)
        p = self.p_bottom * (1.0 - z / self.L_z)   # linear p profile
        M_v = self._m_v(E_1d)
        sigma_prime_z = self.q - self.b * p
        eps_z = sigma_prime_z / M_v
        return float(np.sum(eps_z) * self.dz)

    def _solve_transient(self, E_1d, k_v):
        """
        Forward-Euler time-stepping for dp/dt = c_v * d²p/dz².

        c_v is approximated as k_v * mean(M_v) / gamma_w  where gamma_w = 1 × 10⁴ N/m³
        (unit weight of water).  The stable time step is enforced via the
        Courant condition dt ≤ dz²/(2 c_v).
        """
        M_v = self._m_v(E_1d)
        gamma_w = 1.0e4  # N/m³
        c_v = k_v * float(np.mean(M_v)) / gamma_w  # m²/s

        # Translate non-dimensional T_final to physical time
        c_v_safe = max(c_v, 1.0e-30)
        t_final = self.T_final * self.L_z ** 2 / c_v_safe

        # Maximum stable dt for explicit scheme
        dt_stable = 0.49 * self.dz ** 2 / c_v_safe
        n_steps = max(50, int(np.ceil(t_final / dt_stable)))
        _max_steps = int(self.max_time_steps)
        if n_steps > _max_steps:
            import warnings
            warnings.warn(
                f"BiotSolver1D transient: required {n_steps} time steps for "
                f"stability, but capped at {_max_steps} (solver.max_time_steps). "
                "Physical accuracy may be reduced. Increase max_time_steps or "
                "use a coarser grid to improve fidelity.",
                UserWarning, stacklevel=3,
            )
        n_steps = min(n_steps, _max_steps)
        dt = t_final / n_steps

        # Courant number (must be ≤ 0.5 for stability)
        r = c_v_safe * dt / self.dz ** 2
        r = min(r, 0.49)

        # Initial excess pore pressure = b * q (undrained response)
        p = np.full(self.n_z, self.b * self.q)
        p[0] = self.p_bottom    # bottom BC
        p[-1] = 0.0             # top BC (drain)

        for _ in range(n_steps):
            p_new = p.copy()
            p_new[1:-1] = p[1:-1] + r * (p[2:] - 2.0 * p[1:-1] + p[:-2])
            p_new[0] = self.p_bottom
            p_new[-1] = 0.0
            p = p_new

        sigma_prime_z = self.q - self.b * p
        eps_z = sigma_prime_z / M_v
        return float(np.sum(eps_z) * self.dz)

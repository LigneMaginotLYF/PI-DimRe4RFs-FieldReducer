import yaml
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Allowed values for enum-like config keys
# ---------------------------------------------------------------------------
_ALLOWED = {
    'solver.type': {'1d', '2d'},
    'solver.response_mode': {'steady_state', 'transient'},
    'dimension_reducer.basis_type': {'polynomial', 'kl'},
    'dimension_reducer.mode': {'identity', 'learned'},
    'surrogate.type': {'nn', 'pce'},
}

# Minimum / maximum bounds for numeric config keys
_BOUNDS = {
    'dimension_reducer.d': (1, None),
    'dimension_reducer.basis_order': (1, 10),
    'reduced_lut.n_grid_points': (10, None),
    'solver.n_nodes_x': (1, None),
    'solver.n_nodes_z': (2, None),
    'dataset.n_samples': (2, None),
    'dataset.n_kl_terms_E': (1, None),
    'solver.t_final': (1.0e-6, None),
}


class ConfigManager:
    def __init__(self, config_file=None, preset_file=None):
        """Load and merge configurations from base config and optional preset."""
        self.config = {}
        if config_file and os.path.exists(config_file):
            self.config = self._load_yaml(config_file)
        if preset_file and os.path.exists(preset_file):
            preset = self._load_yaml(preset_file)
            self.config = self._deep_merge(self.config, preset)

    def _load_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        return self._coerce_numeric_strings(raw)

    def _coerce_numeric_strings(self, obj):
        """Recursively convert numeric strings (e.g. '1.0e6') to float/int."""
        if isinstance(obj, dict):
            return {k: self._coerce_numeric_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._coerce_numeric_strings(v) for v in obj]
        if isinstance(obj, str):
            try:
                return int(obj)
            except ValueError:
                pass
            try:
                return float(obj)
            except ValueError:
                pass
        return obj

    def _deep_merge(self, base, override):
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    def get(self, *keys, default=None):
        """Get nested config value using multiple args."""
        d = self.config
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    # ------------------------------------------------------------------
    def validate(self, config=None):
        """
        Comprehensive config validation.

        Parameters
        ----------
        config : dict, optional
            Configuration dict to validate.  Defaults to ``self.config``.

        Checks
        ------
        1. Presence of required top-level sections.
        2. Allowed values for enum-like keys.
        3. Numeric bounds.
        4. Cross-key compatibility constraints.

        Raises
        ------
        ValueError
            With an informative message for every violation detected.
        """
        cfg = config if config is not None else self.config
        errors = []

        # 1. Required sections
        for section in ('dataset', 'material', 'solver'):
            if section not in cfg:
                errors.append(f"Missing required config section: '{section}'")

        def _get(dotted_key, default=None):
            parts = dotted_key.split('.')
            d = cfg
            for p in parts:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    return default
            return d

        # 2. Enum-like keys
        for dotted_key, allowed in _ALLOWED.items():
            val = _get(dotted_key)
            if val is not None and val not in allowed:
                errors.append(
                    f"Config key '{dotted_key}' = {val!r} is not valid. "
                    f"Allowed values: {sorted(allowed)}"
                )

        # 3. Numeric bounds
        for dotted_key, (lo, hi) in _BOUNDS.items():
            val = _get(dotted_key)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                errors.append(f"Config key '{dotted_key}' must be numeric, got {type(val).__name__}")
                continue
            if lo is not None and val < lo:
                errors.append(f"Config key '{dotted_key}' = {val} must be ≥ {lo}")
            if hi is not None and val > hi:
                errors.append(f"Config key '{dotted_key}' = {val} must be ≤ {hi}")

        # 4. Cross-key compatibility
        solver_type = _get('solver.type', '2d')
        n_nodes_x = _get('solver.n_nodes_x', 20)
        response_mode = _get('solver.response_mode', 'steady_state')
        red_mode = _get('dimension_reducer.mode', 'learned')
        basis_type = _get('dimension_reducer.basis_type', 'polynomial')
        d = _get('dimension_reducer.d', 1)
        basis_order = _get('dimension_reducer.basis_order', 1)
        n_kl = _get('dataset.n_kl_terms_E', 5)
        surr_types_list = _get('surrogate.types')
        if surr_types_list is None:
            surr_type_single = _get('surrogate.type', 'nn')
            surr_types_list = [surr_type_single] if surr_type_single else ['nn']
        for st in surr_types_list:
            if st not in _ALLOWED['surrogate.type']:
                errors.append(
                    f"surrogate.types contains unsupported value {st!r}. "
                    f"Allowed: {sorted(_ALLOWED['surrogate.type'])}"
                )

        # 1D solver → n_nodes_x should be 1 (warn if > 1)
        if solver_type == '1d' and n_nodes_x > 1:
            errors.append(
                f"solver.type='1d' but solver.n_nodes_x={n_nodes_x}. "
                "Set n_nodes_x=1 for a purely 1-D problem (single settlement value)."
            )

        # transient mode requires t_final
        if response_mode == 'transient':
            t_final = _get('solver.t_final')
            if t_final is None:
                errors.append(
                    "solver.response_mode='transient' requires solver.t_final to be set "
                    "(non-dimensional final time; e.g. 1.0 for ~95% consolidation)."
                )

        # identity mode: d should equal n_kl (or user accepts partial identity)
        if red_mode == 'identity' and d != n_kl:
            errors.append(
                f"dimension_reducer.mode='identity' should use d = n_kl_terms_E "
                f"(currently d={d}, n_kl_terms_E={n_kl}). "
                "Set d to equal n_kl_terms_E for a true identity mapping."
            )

        # polynomial basis: d must not exceed available basis functions
        if basis_type == 'polynomial':
            # Number of multivariate polynomial basis functions of degree ≤ basis_order
            # for a 2-D physical domain: C(basis_order + 2, 2).
            # order 1 → C(3,2)=3, order 2 → C(4,2)=6, order 3 → C(5,2)=10,
            # order k → (k+1)(k+2)/2. The dict covers common orders exactly.
            def _n_poly_basis(order):
                return (order + 1) * (order + 2) // 2
            n_basis = _n_poly_basis(basis_order)
            if d > n_basis:
                errors.append(
                    f"dimension_reducer.d={d} exceeds the {n_basis} polynomial basis "
                    f"functions available for basis_order={basis_order}. "
                    "Reduce d or increase basis_order."
                )

        # KL basis: d must not exceed n_kl
        if basis_type == 'kl' and d > n_kl:
            errors.append(
                f"dimension_reducer.basis_type='kl' and d={d} > "
                f"n_kl_terms_E={n_kl}. "
                "The reduced KL dimension d cannot exceed the number of KL terms."
            )

        if errors:
            raise ValueError(
                "Configuration validation failed with the following errors:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
        return True


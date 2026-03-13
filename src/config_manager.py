import yaml
import os
from pathlib import Path


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

    def validate(self):
        """Basic validation of required config sections."""
        required = ['dataset', 'material', 'solver']
        for r in required:
            if r not in self.config:
                raise ValueError(f"Missing required config section: {r}")
        return True

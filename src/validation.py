import numpy as np
from sklearn.metrics import r2_score


class Validation:
    @staticmethod
    def compute_metrics(predictions, targets):
        """
        Compute R², RMSE, and relative L² error.

        Args:
            predictions: shape (n_samples, n_x) or (n_x,)
            targets: same shape
        Returns:
            dict with 'r2', 'rmse', 'rel_l2'
        """
        pred = np.atleast_2d(predictions)
        tgt = np.atleast_2d(targets)

        # R² requires at least 2 samples; return None if undefined
        if len(tgt) < 2:
            r2 = None
        else:
            r2_val = r2_score(tgt, pred, multioutput='uniform_average')
            r2 = None if np.isnan(r2_val) else float(r2_val)

        rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))

        norms = np.linalg.norm(tgt, axis=1)
        diff_norms = np.linalg.norm(pred - tgt, axis=1)
        mask = norms > 1e-15
        rel_l2 = float(np.mean(diff_norms[mask] / norms[mask])) if mask.any() else 0.0

        return {'r2': r2, 'rmse': rmse, 'rel_l2': rel_l2}

    @staticmethod
    def compute_per_sample_metrics(predictions, targets):
        """Compute metrics for each sample individually."""
        pred = np.atleast_2d(predictions)
        tgt = np.atleast_2d(targets)
        r2_per = np.array([
            r2_score(tgt[i], pred[i]) for i in range(len(tgt))
        ])
        rmse_per = np.sqrt(np.mean((pred - tgt) ** 2, axis=1))
        norms = np.linalg.norm(tgt, axis=1)
        diff_norms = np.linalg.norm(pred - tgt, axis=1)
        rel_l2_per = np.where(norms > 1e-15, diff_norms / norms, 0.0)
        return {'r2': r2_per, 'rmse': rmse_per, 'rel_l2': rel_l2_per}

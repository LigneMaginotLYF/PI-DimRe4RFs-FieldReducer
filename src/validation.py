import numpy as np
from sklearn.metrics import r2_score

# Threshold below which SS_tot is treated as zero (flat profile guard).
_VARIANCE_THRESHOLD = 1e-30


class Validation:
    @staticmethod
    def compute_metrics(predictions, targets):
        """
        Compute R², RMSE, and relative L² error.

        R² is computed with `variance_weighted` multioutput, which weights each
        output node by its variance across samples.  This down-weights nodes with
        near-zero variance (e.g. a settlement node that barely changes between
        samples), which would otherwise contribute a strongly negative term to the
        simple uniform average and produce a misleadingly negative aggregate R².

        Args:
            predictions: shape (n_samples, n_x) or (n_x,)
            targets: same shape
        Returns:
            dict with 'r2', 'rmse', 'rel_l2'
        """
        pred = np.atleast_2d(predictions)
        tgt = np.atleast_2d(targets)

        # R² requires at least 2 samples; return None if undefined.
        # Use variance_weighted multioutput to down-weight output nodes that have
        # near-zero variance across samples (avoids spuriously negative R²).
        if len(tgt) < 2:
            r2 = None
        else:
            try:
                r2_val = r2_score(tgt, pred, multioutput='variance_weighted')
                r2 = None if np.isnan(r2_val) else float(r2_val)
            except Exception:
                r2 = None

        rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))

        norms = np.linalg.norm(tgt, axis=1)
        diff_norms = np.linalg.norm(pred - tgt, axis=1)
        mask = norms > 1e-15
        rel_l2 = float(np.mean(diff_norms[mask] / norms[mask])) if mask.any() else 0.0

        return {'r2': r2, 'rmse': rmse, 'rel_l2': rel_l2}

    @staticmethod
    def compute_per_sample_metrics(predictions, targets):
        """
        Compute metrics for each sample individually.

        Per-sample R² measures how well the predicted settlement *profile shape*
        matches the true profile for each test sample (treating each spatial node
        as an independent "observation").  It can be negative when the prediction
        is worse than a flat mean line for that particular sample.
        """
        pred = np.atleast_2d(predictions)
        tgt = np.atleast_2d(targets)

        r2_per = np.zeros(len(tgt))
        for i in range(len(tgt)):
            ss_res = np.sum((tgt[i] - pred[i]) ** 2)
            ss_tot = np.sum((tgt[i] - np.mean(tgt[i])) ** 2)
            if ss_tot < _VARIANCE_THRESHOLD:
                r2_per[i] = 1.0 if ss_res < _VARIANCE_THRESHOLD else 0.0
            else:
                r2_per[i] = 1.0 - ss_res / ss_tot

        rmse_per = np.sqrt(np.mean((pred - tgt) ** 2, axis=1))
        norms = np.linalg.norm(tgt, axis=1)
        diff_norms = np.linalg.norm(pred - tgt, axis=1)
        rel_l2_per = np.where(norms > 1e-15, diff_norms / norms, 0.0)
        return {'r2': r2_per, 'rmse': rmse_per, 'rel_l2': rel_l2_per}

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class Visualization:
    def __init__(self, plots_dir='plots'):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        for sub in ['material_fields', 'settlement_comparison', 'sensitivity', 'aggregate']:
            os.makedirs(os.path.join(plots_dir, sub), exist_ok=True)

    def plot_surrogate_comparison(self, metrics_by_type, label='Phase-2 surrogate'):
        """
        Plot side-by-side R² / RMSE comparison for multiple surrogate types.

        Args:
            metrics_by_type: dict  {surrogate_type_str: metrics_dict}
            label: figure title prefix
        """
        plt = self._get_plt()
        types = list(metrics_by_type.keys())
        r2_vals = [metrics_by_type[t].get('r2') or 0.0 for t in types]
        rmse_vals = [metrics_by_type[t].get('rmse') or 0.0 for t in types]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(types))

        axes[0].bar(x, r2_vals, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(types)
        axes[0].set_ylabel('R²')
        axes[0].set_title(f'{label} – R² comparison')
        axes[0].set_ylim(-0.1, 1.05)
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(x, rmse_vals, color='salmon', edgecolor='black', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(types)
        axes[1].set_ylabel('RMSE')
        axes[1].set_title(f'{label} – RMSE comparison')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'aggregate', 'surrogate_comparison.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved surrogate comparison plot to {path}")

    def _get_plt(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt

    def plot_settlement_comparison(self, Y_original, Y_predicted, n_samples=5,
                                    x_positions=None):
        """
        Plot GT vs reduced-space predicted settlement profiles.

        Two curves per panel:
          - Blue solid line  : Ground-truth (GT) settlement profile
          - Red dashed line  : Reduced-space prediction
        The GT curve is drawn with no markers; call
        ``plot_settlement_comparison_with_collocation`` to additionally mark
        collocation x-positions on the GT curve.

        Args:
            Y_original   : Reference settlement profiles, shape (n_test, n_x)
            Y_predicted  : Predicted settlement profiles, shape (n_test, n_x)
            n_samples    : Number of random samples to plot
            x_positions  : Physical x-coordinates, shape (n_x,).  Defaults to
                           0, 1, ..., n_x-1 (node indices).
        """
        plt = self._get_plt()
        n_x = Y_original.shape[1]
        if x_positions is None:
            x_positions = np.arange(n_x)

        n = min(n_samples, len(Y_original))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(Y_original), size=n, replace=False)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            ax.plot(x_positions, Y_original[idx], 'b-', label='GT', linewidth=2)
            ax.plot(x_positions, Y_predicted[idx], 'r--',
                    label='Reduced-space prediction', linewidth=2)
            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('x position')
            ax.set_ylabel('Settlement [m]')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'settlement_comparison', 'comparison.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved settlement comparison plot to {path}")

    def plot_aggregate_metrics(self, Y_original, Y_predicted):
        """Plot R² histogram, RMSE distribution, error vs sample index."""
        from src.validation import Validation
        plt = self._get_plt()
        metrics = Validation.compute_per_sample_metrics(Y_predicted, Y_original)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(metrics['r2'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('R²')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'R² Distribution (mean={np.mean(metrics["r2"]):.3f})')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(metrics['rmse'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('RMSE [m]')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'RMSE Distribution (mean={np.mean(metrics["rmse"]):.3e})')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(metrics['rel_l2'], 'o', markersize=4, color='purple', alpha=0.6)
        axes[2].set_xlabel('Sample index')
        axes[2].set_ylabel('Relative L² error')
        axes[2].set_title(f'Relative L² (mean={np.mean(metrics["rel_l2"]):.3f})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'aggregate', 'metrics.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved aggregate metrics plot to {path}")

    def plot_material_fields(self, E_fields, E_reduced_values=None,
                             k_h_values=None, k_v_values=None, n_samples=5):
        """
        Compare original E fields with reduced (constant) E values.

        Three-row layout:
          Row 0 – Original KL-expanded E field (heatmap)
          Row 1 – Reduced constant E' field (heatmap, same colour scale)
          Row 2 – Difference / error map  |E_orig - E_reduced|

        k_h_values / k_v_values: if provided, the reduced permeability constants
        are annotated on the reduced-field panel.
        """
        plt = self._get_plt()
        n = min(n_samples, len(E_fields))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(E_fields), size=n, replace=False)

        n_rows = 3 if E_reduced_values is not None else 1
        fig, axes = plt.subplots(n_rows, n, figsize=(4 * n, 4 * n_rows))

        # Ensure axes is always 2-D
        if n_rows == 1 and n == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n == 1:
            axes = axes[:, np.newaxis]

        for col, i in enumerate(sample_idx):
            E_orig = E_fields[i]
            vmin = E_orig.min()
            vmax = E_orig.max()

            # Row 0: Original E field
            im0 = axes[0, col].imshow(E_orig, aspect='auto', cmap='viridis',
                                      vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
            axes[0, col].set_title(f'Original E field\nsample {i}')
            axes[0, col].set_xlabel('x node')
            axes[0, col].set_ylabel('z node')

            if E_reduced_values is not None and n_rows >= 2:
                E_val_input = E_reduced_values[i]
                if isinstance(E_val_input, np.ndarray):
                    E_red = E_val_input  # pre-reconstructed 2D field
                else:
                    E_red = np.full_like(E_orig, float(E_val_input))

                # Row 1: Reduced constant E' field
                im1 = axes[1, col].imshow(E_red, aspect='auto', cmap='viridis',
                                          vmin=vmin, vmax=vmax)
                plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
                if isinstance(E_val_input, np.ndarray):
                    E_mean = float(np.mean(E_val_input))
                    subtitle = f"Reduced E' (mean={E_mean:.2e})"
                else:
                    subtitle = f"Reduced E'={float(E_val_input):.2e}"
                if k_h_values is not None:
                    subtitle += f"\nk_h={k_h_values[i]:.2e}"
                if k_v_values is not None:
                    subtitle += f", k_v={k_v_values[i]:.2e}"
                axes[1, col].set_title(subtitle)
                axes[1, col].set_xlabel('x node')
                axes[1, col].set_ylabel('z node')

                # Row 2: Difference / error map
                diff = np.abs(E_orig - E_red)
                im2 = axes[2, col].imshow(diff, aspect='auto', cmap='Reds')
                plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)
                axes[2, col].set_title(f'|E_orig - E_reduced|\nmax={diff.max():.2e}')
                axes[2, col].set_xlabel('x node')
                axes[2, col].set_ylabel('z node')

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'material_fields', 'comparison.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved material fields plot to {path}")

    def plot_sobol_sensitivity(self, reducer, input_dim=5, n_samples=1000):
        """
        Approximate Sobol first-order sensitivity indices for the reducer M.
        Shows which xi_E components most affect the xi' output.
        """
        plt = self._get_plt()
        rng = np.random.default_rng(42)

        A = rng.standard_normal((n_samples, input_dim))
        B = rng.standard_normal((n_samples, input_dim))

        if hasattr(reducer, 'predict'):
            fA = reducer.predict(A)
            fB = reducer.predict(B)
        else:
            import torch
            with torch.no_grad():
                fA = reducer(torch.tensor(A, dtype=torch.float32)).numpy()
                fB = reducer(torch.tensor(B, dtype=torch.float32)).numpy()

        output_dim = fA.shape[1]
        S1 = np.zeros((input_dim, output_dim))

        for k in range(input_dim):
            AB_k = A.copy()
            AB_k[:, k] = B[:, k]
            if hasattr(reducer, 'predict'):
                f_AB_k = reducer.predict(AB_k)
            else:
                import torch
                with torch.no_grad():
                    f_AB_k = reducer(torch.tensor(AB_k, dtype=torch.float32)).numpy()
            var_Y = np.var(fA, axis=0) + 1e-15
            S1[k] = 1 - np.mean((fB - f_AB_k) ** 2, axis=0) / (2 * var_Y)

        S1_mean = np.mean(np.clip(S1, 0, 1), axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(input_dim)
        ax.bar(x, S1_mean, color='teal', edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'ξ_E[{i}]' for i in range(input_dim)])
        ax.set_ylabel('First-order Sobol index (approx.)')
        ax.set_title('Sensitivity: Which KL terms drive the reduced parameters?')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'sensitivity', 'sobol_indices.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Sobol sensitivity plot to {path}")

    @staticmethod
    def plot_material_fields_static(fields):
        """Static method for backward compatibility."""
        viz = Visualization()
        viz.plot_material_fields(fields)

    def plot_settlement_comparison_with_collocation(self, Y_original, Y_predicted,
                                                     reduced_lut, n_samples=5,
                                                     x_positions=None,
                                                     colloc_idx=None):
        """
        Plot GT vs reduced-space settlement profiles with collocation positions marked.

        Layout per panel:
          - Blue solid line       : GT settlement profile
          - Red dashed line       : Reduced-space prediction
          - Green circle markers  : GT values at collocation x-positions
            (marks WHERE Phase-3 collocation constraints are applied)

        Args:
            Y_original    : shape (n_test, n_x)
            Y_predicted   : shape (n_test, n_x)
            reduced_lut   : ReducedLUT with train_indices and responses attributes
            n_samples     : Number of random test samples to plot
            x_positions   : Physical x-coordinates of ALL n_x output nodes,
                            shape (n_x,).  Defaults to 0, 1, ..., n_x-1.
                            Must match the full n_x width of Y_original.
            colloc_idx    : 1-D integer array of node indices used as collocation
                            points in Phase-3 training.  Green circles are drawn
                            only at these positions.  If None, circles are drawn
                            at every node (all nodes are collocation points).
        """
        plt = self._get_plt()
        n_x = Y_original.shape[1]
        if x_positions is None:
            x_positions = np.arange(n_x)

        # Ensure x_positions matches n_x (full grid), not collocation subset
        if len(x_positions) != n_x:
            x_positions = np.linspace(x_positions[0], x_positions[-1], n_x)

        # Collocation x-positions for markers
        if colloc_idx is not None:
            # Guard against indices exceeding current n_x
            colloc_idx = colloc_idx[colloc_idx < n_x]
            colloc_x = x_positions[colloc_idx]
        else:
            colloc_x = x_positions
            colloc_idx = np.arange(n_x)

        n = min(n_samples, len(Y_original))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(Y_original), size=n, replace=False)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            # GT curve (full profile)
            ax.plot(x_positions, Y_original[idx], 'b-',
                    label='GT', linewidth=2.5, zorder=3)
            # Reduced-space prediction curve (full profile)
            ax.plot(x_positions, Y_predicted[idx], 'r--',
                    label='Reduced-space prediction', linewidth=2.5, zorder=3)
            # Collocation positions marked as circles ON the GT curve
            ax.plot(colloc_x, Y_original[idx][colloc_idx], 'go',
                    markersize=6, markerfacecolor='none', markeredgewidth=1.5,
                    label='Collocation positions', zorder=4)

            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('x position')
            ax.set_ylabel('Settlement [m]')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'settlement_comparison',
                            'comparison_with_collocation.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved settlement comparison with collocation positions to {path}")

    def plot_phase2_surrogate_accuracy(self, reduced_lut, surrogate_type='nn',
                                        val_fraction=0.2, seed=42):
        """
        Generate and save accuracy plots for the Phase-2 forward surrogate.

        Plots:
          1. Scatter: predicted vs. actual settlement for all output nodes
             (validation set)
          2. A few example settlement profiles: surrogate vs. direct solver

        Saved under ``<plots_dir>/phase2_surrogate/``.

        Args:
            reduced_lut    : Fitted ReducedLUT with surrogate loaded.
            surrogate_type : 'nn' or 'pce'.
            val_fraction   : Fraction of LUT data used as validation.
            seed           : Random seed for train/val split.
        """
        plt = self._get_plt()
        rng = np.random.default_rng(seed)

        n = len(reduced_lut.grid_points)
        idx = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = idx[:n_val]

        X_val = reduced_lut.grid_points[val_idx]
        Y_val = reduced_lut.responses[val_idx]
        Y_pred_val = reduced_lut.predict(X_val)

        surr_dir = os.path.join(self.plots_dir, 'phase2_surrogate')
        os.makedirs(surr_dir, exist_ok=True)

        # --- Plot 1: predicted vs. actual scatter ---
        fig, ax = plt.subplots(figsize=(6, 6))
        y_flat = Y_val.ravel()
        yp_flat = Y_pred_val.ravel()
        ax.scatter(y_flat, yp_flat, alpha=0.3, s=5, color='steelblue')
        lo, hi = min(y_flat.min(), yp_flat.min()), max(y_flat.max(), yp_flat.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('True settlement [m]')
        ax.set_ylabel('Surrogate prediction [m]')
        ax.set_title(f'Phase-2 surrogate ({surrogate_type}) – scatter (n_val={len(X_val)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(surr_dir, f'scatter_{surrogate_type}.png')
        plt.savefig(scatter_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Phase-2 surrogate scatter to {scatter_path}")

        # --- Plot 2: example profiles ---
        n_ex = min(5, len(X_val))
        ex_idx = rng.choice(len(X_val), size=n_ex, replace=False)
        x_pos = np.arange(Y_val.shape[1])

        fig, axes = plt.subplots(1, n_ex, figsize=(4 * n_ex, 4))
        if n_ex == 1:
            axes = [axes]
        for ax, i in zip(axes, ex_idx):
            ax.plot(x_pos, Y_val[i], 'b-', label='Direct solver', linewidth=2)
            ax.plot(x_pos, Y_pred_val[i], 'r--', label=f'Surrogate ({surrogate_type})',
                    linewidth=2)
            ax.set_xlabel('x position')
            ax.set_ylabel('Settlement [m]')
            ax.set_title(f'LUT point {val_idx[i]}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.0)
        plt.tight_layout()
        profiles_path = os.path.join(surr_dir, f'profiles_{surrogate_type}.png')
        plt.savefig(profiles_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Phase-2 surrogate profiles to {profiles_path}")



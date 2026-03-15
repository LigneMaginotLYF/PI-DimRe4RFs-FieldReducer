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

    def plot_settlement_comparison(self, Y_original, Y_predicted, n_samples=5):
        """Plot original vs predicted settlement profiles for n_samples."""
        plt = self._get_plt()
        n = min(n_samples, len(Y_original))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(Y_original), size=n, replace=False)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            x = np.arange(Y_original.shape[1])
            ax.plot(x, Y_original[idx], 'b-', label='Original', linewidth=2)
            ax.plot(x, Y_predicted[idx], 'r--', label='Reduced', linewidth=2)
            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('x node')
            ax.set_ylabel('Settlement [m]')
            ax.legend()
            ax.grid(True, alpha=0.3)

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
                                                     reduced_lut, n_samples=5):
        """
        Plot original vs predicted settlement profiles with collocation points overlay.

        Args:
            Y_original: Reference settlement profiles, shape (n_test, n_x)
            Y_predicted: Predicted settlement profiles, shape (n_test, n_x)
            reduced_lut: ReducedLUT object with grid_points, responses, train_indices
            n_samples: Number of samples to plot
        """
        plt = self._get_plt()
        n = min(n_samples, len(Y_original))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(Y_original), size=n, replace=False)

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        train_indices = reduced_lut.train_indices
        collocation_settlements = reduced_lut.responses[train_indices]

        for ax, idx in zip(axes, indices):
            x = np.arange(Y_original.shape[1])

            ax.plot(x, Y_original[idx], 'b-', label='Original', linewidth=2.5)
            ax.plot(x, Y_predicted[idx], 'r--', label='Reduced', linewidth=2.5)

            n_colloc_show = min(20, len(collocation_settlements))
            colloc_idx = rng.choice(len(collocation_settlements),
                                    size=n_colloc_show, replace=False)
            for c_idx in colloc_idx:
                ax.plot(x, collocation_settlements[c_idx], 'go', alpha=0.3, markersize=4)

            ax.plot([], [], 'go', alpha=0.5, markersize=6,
                    label='Collocation points (training)')
            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('x node')
            ax.set_ylabel('Settlement [m]')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'settlement_comparison',
                            'comparison_with_collocation.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved settlement comparison with collocation points to {path}")

    def plot_settlement_collocation_scatter(self, reduced_lut):
        """
        Create scatter plot of collocation point coverage in settlement space.

        Shows spatial position (x node) vs settlement value for all training
        collocation points.  Helps visualize the physical constraints used when
        training the surrogate.

        Args:
            reduced_lut: ReducedLUT object with responses and train_indices
        """
        plt = self._get_plt()

        train_indices = reduced_lut.train_indices
        collocation_settlements = reduced_lut.responses[train_indices]

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, y_profile in enumerate(collocation_settlements):
            x_nodes = np.arange(len(y_profile))
            ax.scatter(x_nodes, y_profile, alpha=0.4, s=20,
                       c=np.full(len(y_profile), i), cmap='viridis',
                       vmin=0, vmax=len(collocation_settlements))

        ax.set_xlabel('x node')
        ax.set_ylabel('Settlement [m]')
        ax.set_title(
            f'Collocation Points in Settlement Space '
            f'(n={len(collocation_settlements)} training points)'
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, 'settlement_comparison',
                            'collocation_scatter.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved collocation scatter plot to {path}")

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

    def plot_material_fields(self, E_fields, E_reduced_values=None, n_samples=5):
        """Compare original E fields with reduced (constant) E values."""
        plt = self._get_plt()
        n = min(n_samples, len(E_fields))
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

        for i in range(n):
            im = axes[0, i].imshow(E_fields[i], aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=axes[0, i])
            axes[0, i].set_title(f'Original E field {i}')
            axes[0, i].set_xlabel('x node')
            axes[0, i].set_ylabel('z node')

            if E_reduced_values is not None:
                E_red = np.full_like(E_fields[i], E_reduced_values[i])
                im2 = axes[1, i].imshow(E_red, aspect='auto', cmap='viridis',
                                        vmin=E_fields[i].min(), vmax=E_fields[i].max())
                plt.colorbar(im2, ax=axes[1, i])
                axes[1, i].set_title(f'Reduced E (const={E_reduced_values[i]:.2e})')
            else:
                axes[1, i].axis('off')

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

"""
scripts/train_phase2.py – Decoupled Phase-1 + Phase-2 training script.

Runs only Phase 1 (dataset generation) and Phase 2 (LUT surrogate training)
and saves all Phase-2 artifacts without proceeding to Phase 3 / 4.
This is useful when you want to iterate quickly on the surrogate architecture
or output representation without waiting for the full four-phase run.

Usage
-----
    python scripts/train_phase2.py [--config CONFIG] [--preset PRESET]
                                   [--output-dir OUTPUT_DIR] [--seed SEED]

Arguments
---------
--config      Path to base config YAML (default: config.yaml).
--preset      Path to a preset YAML that overrides the base config (optional).
--output-dir  Directory for run artefacts (default: auto-timestamped under results/).
--seed        Override dataset.seed from config (optional integer).

Outputs
-------
The following artefacts are written under the output directory and the shared
``data/`` / ``models/`` directories (same paths as the full four-phase pipeline):

  data/
    X_train.npy           -- Phase-1 coefficient matrix  (n_samples, n_features)
    Y_train.npy           -- Phase-1 settlement profiles (n_samples, n_nodes_x)
    lut_grid_points.npy   -- LUT grid in xi'-space       (n_grid, effective_d)
    lut_responses.npy     -- LUT Biot responses          (n_grid, n_nodes_x)

  models/reduced_lut/
    grid_points.npy       -- same as lut_grid_points.npy
    responses.npy         -- same as lut_responses.npy
    surrogate_*.pt / .pkl -- fitted surrogate model
    config.json           -- surrogate metadata and config hash
    <type>/evaluation/    -- independent test-set metrics and plots
"""

import argparse
import logging
import os
import sys

# Allow running from the repository root without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run Phase 1 (dataset generation) + Phase 2 (LUT surrogate training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to the base configuration YAML file.',
    )
    parser.add_argument(
        '--preset', default=None,
        help='Path to a preset YAML that overrides the base config (optional).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for run artefacts.  Defaults to an auto-timestamped folder '
             'under results/.',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override dataset.seed from config.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    from src.config_manager import ConfigManager
    cm = ConfigManager(config_file=args.config, preset_file=args.preset)
    config = cm.config

    if args.seed is not None:
        config.setdefault('dataset', {})['seed'] = args.seed

    from src.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline(config, output_dir=args.output_dir)

    logger.info("Starting Phase-1 dataset generation …")
    X_train, Y_train = pipeline.phase1_generate_dataset()
    logger.info(
        f"Phase 1 complete: X_train={X_train.shape}, Y_train={Y_train.shape}"
    )

    logger.info("Starting Phase-2 LUT surrogate training …")
    lut = pipeline.phase2_build_reduced_surrogate()
    logger.info(
        f"Phase 2 complete: LUT grid={lut.grid_points.shape}, "
        f"surrogate={type(lut.surrogate).__name__}"
    )

    logger.info("Phase-1 and Phase-2 artefacts saved.  Exiting (Phase 3/4 not run).")


if __name__ == '__main__':
    main()

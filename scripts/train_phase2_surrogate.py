"""
scripts/train_phase2_surrogate.py – Standalone Phase-2 surrogate training script.

Alias for ``scripts/train_phase2.py``.  Runs Phase-1 dataset generation and
Phase-2 LUT surrogate training, saves all artefacts, and exits **without**
running Phase 3 or Phase 4.

After this script completes you can run Phase 3 + 4 with:

    python train.py --phases 3,4 \\
        --config config.yaml \\
        --output-dir <same-output-dir-as-phase2>

Make sure ``phase3.surrogate_type_to_use`` in your config matches the
``phase2.surrogate_type`` used here so Phase-3 loads the correct artifact.

Usage
-----
    python scripts/train_phase2_surrogate.py [--config CONFIG] [--preset PRESET]
                                             [--output-dir OUTPUT_DIR] [--seed SEED]
                                             [--surrogate-type {nn,pce}]
                                             [--output-repr {direct,dct,poly,bspline}]

Arguments
---------
--config          Path to base config YAML (default: config.yaml).
--preset          Path to a preset YAML that overrides the base config (optional).
--output-dir      Directory for run artefacts (default: auto-timestamped under results/).
--seed            Override dataset.seed from config (optional integer).
--surrogate-type  Override phase2.surrogate_type: "nn" or "pce".
--output-repr     Override surrogate.output_representation: direct, dct, poly, or bspline.

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

Notes
-----
- Supports all output representations: 'direct', 'dct', 'poly', 'bspline'.
- After training, Phase 3 can be run independently by pointing it at the saved
  artefacts via ``phase3.surrogate_type_to_use`` in config.yaml.
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
        description=(
            'Run Phase 1 (dataset generation) + Phase 2 (LUT surrogate training). '
            'Phase 3 and 4 are NOT run.  Use train.py --phases 3,4 afterwards.'
        ),
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
    parser.add_argument(
        '--surrogate-type', choices=['nn', 'pce'], default=None,
        help='Override phase2.surrogate_type: "nn" (neural network) or "pce" '
             '(polynomial chaos expansion).',
    )
    parser.add_argument(
        '--output-repr', choices=['direct', 'dct', 'poly', 'bspline'], default=None,
        dest='output_repr',
        help='Override surrogate.output_representation.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    from src.config_manager import ConfigManager
    cm = ConfigManager(config_file=args.config, preset_file=args.preset)
    config = cm.config

    if args.seed is not None:
        config.setdefault('dataset', {})['seed'] = args.seed
    if args.surrogate_type is not None:
        config.setdefault('phase2', {})['surrogate_type'] = args.surrogate_type
    if args.output_repr is not None:
        config.setdefault('surrogate', {})['output_representation'] = args.output_repr

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
        f"surrogate={type(lut.surrogate).__name__}, "
        f"output_representation={lut.output_representation!r}"
    )

    p2_surr_type = (config.get('phase2') or {}).get('surrogate_type', 'nn')
    logger.info(
        f"\nPhase-1 and Phase-2 artefacts saved.  Exiting (Phase 3/4 not run).\n"
        f"To run Phase 3 + 4 against this surrogate:\n"
        f"  python train.py --phases 3,4 --config {args.config}\n"
        f"  (ensure phase3.surrogate_type_to_use: \"{p2_surr_type}\" in config)"
    )


if __name__ == '__main__':
    main()

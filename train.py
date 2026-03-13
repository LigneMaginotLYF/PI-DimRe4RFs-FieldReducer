import argparse
import logging
import os
import sys


def setup_logging(output_dir=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(output_dir, 'train.log'), encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description='PI-DimRe4RFs-FieldReducer: Four-phase material field dimension reduction'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to base configuration file (default: config.yaml)')
    parser.add_argument('--preset', type=str, default=None,
                        help='Path to preset YAML file (overrides config values)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: results/<timestamp>)')
    parser.add_argument('--phases', type=str, default='1,2,3,4',
                        help='Comma-separated list of phases to run (default: 1,2,3,4)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Override number of training samples')
    parser.add_argument('--surrogate-type', type=str, default=None,
                        choices=['nn', 'pce'], help='Surrogate type (nn or pce)')
    args = parser.parse_args()

    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("PI-DimRe4RFs-FieldReducer starting")
    logger.info(f"Config: {args.config}, Preset: {args.preset}")

    from src.config_manager import ConfigManager
    cfg_mgr = ConfigManager(config_file=args.config, preset_file=args.preset)
    config = cfg_mgr.config

    if args.n_samples is not None:
        config.setdefault('dataset', {})['n_samples'] = args.n_samples
    if args.surrogate_type is not None:
        config.setdefault('surrogate', {})['type'] = args.surrogate_type

    phases = [int(p.strip()) for p in args.phases.split(',')]
    logger.info(f"Running phases: {phases}")

    from src.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline(config, output_dir=args.output_dir)
    metrics = pipeline.orchestrate(phases=phases)

    if metrics:
        logger.info(f"Final test metrics: {metrics}")
    logger.info("Done.")


if __name__ == '__main__':
    main()

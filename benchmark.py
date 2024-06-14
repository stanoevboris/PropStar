import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from rdm.dataset_processor import DatasetProcessor, DatasetConfig
from utils import load_yaml_config


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.config_file = 'datasets.yaml'
        self.initialize_logging()

    @staticmethod
    def initialize_logging():
        logging.basicConfig(
            format='[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S',
            level=logging.INFO)

    def load_datasets(self):
        # Load datasets configuration from YAML or similar
        config = load_yaml_config(self.config_file)

        for dataset in config['datasets']:
            if dataset.get('enabled', True):
                yield dataset

    def check_prop_methods(self):
        if self.args.prop_methods is None:
            raise ValueError('No propositional methods specified')

    def run(self):
        self.check_prop_methods()
        for dataset_info in self.load_datasets():
            processor = DatasetProcessor(dataset_info, self.args)
            processor.process()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="results/experiments_21_05_2024.csv", help="Path to the results file")
    parser.add_argument("--classifier_config", default="classifier_config.yaml",
                        help="Path to the classifiers dataset_config file")
    parser.add_argument("--fe_config", default="fe_config.yaml",
                        help="Path to the feature engineering transformers config file")
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument(
        "--prop_methods", nargs='*', default=['wordification'],
        choices=["wordification", "denormalization"])
    return parser.parse_args()


def main():
    args = parse_arguments()
    benchmark = Benchmark(args)
    benchmark.run()


if __name__ == "__main__":
    main()

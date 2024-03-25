import argparse
import yaml
import subprocess
from itertools import product


def generate_combinations(params):
    """
    Generate all combinations of parameter values.

    Parameters:
    - params: A dictionary where keys are parameter names and values are lists of parameter values.

    Returns:
    - A list of dictionaries, each representing a unique combination of parameter values.
    """
    keys = params.keys()
    values_combinations = list(product(*params.values()))
    return [dict(zip(keys, values)) for values in values_combinations]


def run_hyperparameter_tuning_yaml(config_file, results_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    for classifier in config['classifiers']:
        name = classifier['name']
        for param_set in classifier['params']:
            # Generate all combinations of parameter values for the current set
            combinations = generate_combinations(param_set)

            for combo in combinations:
                cmd = f"python benchmark.py --learner {name} --results_file {results_file}"
                for param, value in combo.items():
                    # For list-type parameters, join the values with a comma
                    if isinstance(value, list):
                        value = ','.join(map(str, value))
                    cmd += f" --{param} {value}"
                print(f"Executing: {cmd}")
                subprocess.run(cmd, shell=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning with specified YAML configuration.')
    parser.add_argument('--config', default='learner_config.yaml', type=str, required=False,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--results_file', default='experiments.csv', type=str, required=False,
                        help='Path to the Results file.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_hyperparameter_tuning_yaml(args.config, args.results_file)

import os
import csv
import yaml
from collections import OrderedDict, Counter

import numpy as np
import logging

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from itertools import product
import constants
PROJECT_DIR = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def get_all_columns():
    # Initialize a set to collect all unique keys
    all_columns = set(constants.COMMON_PARAMS.keys())

    # Add keys from each classifier's parameters in CLASSIFIER_GRID
    for classifier_params in constants.CLASSIFIER_GRID.values():
        all_columns.update(classifier_params.keys())

    # Create a new dictionary with None as default values for each key
    all_columns = {key: None for key in all_columns}
    return all_columns


def setup_directory(directory_path):
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory {directory_path} created.")


def load_yaml_config(config_file: str):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error("datasets.yaml file not found.")
        return


def generate_classifier_params(config_file: str):
    config = load_yaml_config(config_file)

    for classifier in config['classifiers']:
        for param_set in classifier['params']:
            for combination in generate_classifier_combinations(param_set):
                yield classifier['name'], combination


def generate_classifier_combinations(params):
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


def log_dataset_info(train_features, test_features):
    """
    Logs information about the dataset, working with both DataFrames and sparse matrices.
    """
    num_train_records, num_train_features = train_features.shape
    num_test_records, num_test_features = test_features.shape

    total_records = num_train_records + num_test_records

    logging.info(f"Dataset number of records: {total_records}")
    logging.info(f"Train set number of features: {num_train_features}")
    logging.info(f"Test set number of features: {num_test_features}")


def calculate_positive_class_percentage(train_classes, test_classes, representation_type):
    """
    Calculate the percentage occurrence of the positive class in the combined class list or array,
    handling both DataFrames and numpy arrays. The positive class is defined by the 'positive_label'.

    Parameters:
    - train_classes: Training class labels.
    - test_classes: Test class labels.
    - representation_type: Type of representation, affecting how data is processed.

    Returns:
    - The percentage of occurrences of the positive class.
    """
    if representation_type == 'woe':
        # Handling DataFrame: Extracting values as a list
        all_values = list(train_classes.values()) + list(test_classes.values())
    else:
        # Handling numpy array (for sparse matrix, ensure conversion to numpy array beforehand)
        all_values = np.concatenate((train_classes, test_classes))

    # Count occurrences of each class
    occurrences = Counter(all_values)

    # Calculate total number of occurrences
    total_count = sum(occurrences.values())

    # Calculate the percentage of the positive class
    positive_class_count = occurrences.get(1, 0)
    positive_class_percentage = (positive_class_count / total_count) * 100 if total_count > 0 else 0

    return positive_class_percentage


def clear(stx):
    """
    Clean the unneccesary parenthesis
    """

    return stx.replace("`", "").replace("`", "")


def discretize_candidates(df, types, ratio_threshold=0.20, n_bins=20):
    """
    Continuous variables are discretized if more than 30% of the rows are unique.
    """

    ratio_storage = {}
    for enx, type_var in enumerate(types):
        if "int" in type_var or "decimal" in type_var or "float" in type_var:
            ratio_storage = 1. * df[enx].nunique() / df[enx].count()
            if ratio_storage > ratio_threshold and ratio_storage != 1.0:
                to_validate = df[enx].values
                parsed_array = np.array(
                    [np.nan if x == "NULL" else float(x) for x in to_validate])
                parsed_array = interpolate_nans(parsed_array.reshape(-1, 1))
                to_be_discretized = parsed_array.reshape(-1, 1)
                var = KBinsDiscretizer(
                    encode="ordinal",
                    n_bins=n_bins).fit_transform(to_be_discretized)
                df[enx] = var
                if np.isnan(var).any():
                    continue  ## discretization fail
                df[enx] = df[enx].astype(str)  ## cast back to str.
    return df


class OrderedDictList(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class OrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = {}
        return self[k]


def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)


def cleanp(stx):
    """
    Simple string cleaner
    """

    return stx.replace("(", "").replace(")", "").replace(",", "")


def interpolate_nans(X):
    """
    Simply replace nans with column means for numeric variables.
    input: matrix X with present nans
    output: a filled matrix X
    """

    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:, j])
        X[mask_j, j] = np.mean(np.flatnonzero(X))
    return X


def calculate_stats(metric_name, scores):
    stats = {
        f'min_score_{metric_name}': min(scores),
        f'max_score_{metric_name}': max(scores),
        f'mean_score_{metric_name}': np.mean(scores),
        f'std_score_{metric_name}': np.std(scores),
    }
    return stats


def save_results(args, scores, grid_dict):
    results_dir_path = os.path.join(PROJECT_DIR, 'results')
    os.makedirs(results_dir_path, exist_ok=True)
    learner_results_file_path = os.path.join(results_dir_path, args.results_file)

    # Convert args to dictionary and filter relevant keys from grid_dict
    args_dict = vars(args).copy()
    all_columns = get_all_columns()
    # Combine all dictionaries: args_dict, classifier_params, scores, and additional metrics
    combined_dict = {**all_columns,
                     'labels_occurrence_percentage': grid_dict.get('labels_occurrence_percentage', '/'),
                     'total_execution_time': grid_dict.get('total_execution_time', '/')}

    # Calculate stats for each metric and merge them into combined_dict
    for metric in scores:
        metric_stats = calculate_stats(metric_name=metric, scores=scores[metric])
        combined_dict.update(metric_stats)

    combined_dict.update(args_dict)
    combined_dict.update(grid_dict)

    combined_dict = {key: value if value is not None else '/' for key, value in combined_dict.items()}

    # Write to CSV
    with open(learner_results_file_path, "a", newline='') as csvfile:
        file_empty_check = os.stat(learner_results_file_path).st_size == 0
        headers = list(combined_dict.keys())
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=headers)

        if file_empty_check:
            writer.writeheader()

        writer.writerow(combined_dict)


def clean_dataframes(tables: dict):
    for table_name, df in tables.items():
        # remove all columns with null values percentage above 80%
        tables[table_name] = df.loc[:, df.isnull().mean() < .8]

    return tables


def preprocess_tables(target_schema: str, tables: dict) -> dict:
    tables = clean_dataframes(tables=tables)
    if target_schema == 'Sales':
        soh = tables['SalesOrderHeader'].copy()
        soh['previous_order_date'] = soh.groupby('CustomerID')['OrderDate'].shift(1)
        soh['days_without_order'] = (soh['OrderDate'] - soh['previous_order_date']).dt.days.fillna(0)
        cut_off_date = soh['OrderDate'].max() - pd.DateOffset(days=180)

        def calculate_churn(row):
            if row['OrderDate'] >= cut_off_date:
                return None
            elif row['days_without_order'] <= 180:
                return 0
            else:
                return 1

        soh['churn'] = soh.apply(calculate_churn, axis=1)

        # Reset the index
        soh = soh[soh['churn'].notna()]
        soh.reset_index(drop=True, inplace=True)
        soh['churn'] = soh['churn'].astype(np.int64)
        # soh['SalesPersonID'] = soh['churn'].astype(np.int64)
        soh.drop(['previous_order_date', 'days_without_order'], axis=1, inplace=True)

        tables['SalesOrderHeader'] = soh.copy()
    elif target_schema == 'imdb_ijs':
        from imdb_movies_constants import top_250_movies, bottom_100_movies
        movies = tables['movies'].copy()
        positive_df = pd.DataFrame(top_250_movies, columns=["name", "year", "label"])
        negative_df = pd.DataFrame(bottom_100_movies, columns=["name", "year", "label"])
        result_df = pd.concat([positive_df, negative_df], ignore_index=True)
        result_df["year"] = result_df["year"].astype(int)
        result_with_original_data = movies.merge(result_df, on=["name", "year"], how="inner")
        movies = result_with_original_data[["id", "name", "year", "label"]]
        tables["movies"] = movies.copy()
    return tables

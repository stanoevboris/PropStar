import os
import csv
from collections import OrderedDict
import numpy as np
import logging
from sklearn.preprocessing import KBinsDiscretizer

PROJECT_DIR = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


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


def save_results(args, dataset, accuracies, roc_auc_scores, grid_dict):
    results_dir_path = os.path.join(PROJECT_DIR, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    dataset = dataset.split(".")[0]

    learner_results_file_path = os.path.join(results_dir_path, f'{dataset}_{args.representation_type}.csv')
    accuracy_stats = calculate_stats(metric_name='acc', scores=accuracies)
    roc_auc_stats = calculate_stats(metric_name='roc_auc', scores=roc_auc_scores)
    args_dict = {arg: getattr(args, arg) if arg in grid_dict else '/' for arg in vars(args)}
    data_dict = args_dict | accuracy_stats | roc_auc_stats
    with open(learner_results_file_path, "a") as csvfile:
        file_empty_check = os.stat(learner_results_file_path).st_size == 0
        headers = [key for key in data_dict.keys()]
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        if file_empty_check:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(data_dict)

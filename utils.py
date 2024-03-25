import os
import csv
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import logging

import pandas as pd
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


def save_results(args, dataset, scores, grid_dict):
    results_dir_path = os.path.join(PROJECT_DIR, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    learner_results_file_path = os.path.join(results_dir_path, args.results_file)
    args_dict = {arg: getattr(args, arg) if arg in grid_dict else '/' for arg in vars(args)}
    data_dict = args_dict | {'labels_occurrence_percentage': grid_dict.get('labels_occurrence_percentage'),
                             'total_execution_time': grid_dict.get('total_execution_time')}
    for metric in scores:
        data_dict = data_dict | calculate_stats(metric_name=metric, scores=scores[metric])
    with open(learner_results_file_path, "a") as csvfile:
        file_empty_check = os.stat(learner_results_file_path).st_size == 0
        headers = [key for key in data_dict.keys()]
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        if file_empty_check:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(data_dict)


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
        # TODO: drop previous order date and days_without order

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

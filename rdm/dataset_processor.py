import logging
import os
import time

import numpy as np
import pandas as pd

from rdm.db_utils import get_database
from rdm.ml_exeriments import MLExperiment
from rdm.propositionalization import PropConfig, Wordfication, Denormalization

from sklearn.preprocessing import LabelEncoder


class DatasetConfig:
    def __init__(self, dataset_info):
        self.sql_type = dataset_info['sql_type']
        self.database = dataset_info['database']
        self.target_schema = dataset_info['target_schema']
        self.target_table = dataset_info['target_table']
        self.target_attribute = dataset_info['target_column']
        self.include_all_schemas = dataset_info['include_all_schemas']
        self.max_depth = 2


class DatasetProcessor:
    def __init__(self, dataset_info, args):
        self.dataset_config = DatasetConfig(dataset_info)
        self.args = args
        self.initialize_logging()
        self.tables = None
        self.primary_keys = None
        self.foreign_keys = None

    @staticmethod
    def initialize_logging():
        logging.basicConfig(
            format='[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S',
            level=logging.INFO)

    @staticmethod
    def clean_dataframes(tables: dict):
        for table_name, df in tables.items():
            # remove all columns with null values percentage above 80%
            tables[table_name] = df.loc[:, df.isnull().mean() < .8]

        return tables

    def preprocess_tables(self) -> None:
        self.tables = self.clean_dataframes(tables=self.tables)
        if self.dataset_config.target_schema == 'Sales':
            soh = self.tables['SalesOrderHeader'].copy()
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

            self.tables['SalesOrderHeader'] = soh.copy()
        elif self.dataset_config.target_schema == 'imdb_ijs':
            from rdm.imdb_movies_constants import top_250_movies, bottom_100_movies
            movies = self.tables['movies'].copy()
            positive_df = pd.DataFrame(top_250_movies, columns=["name", "year", "label"])
            negative_df = pd.DataFrame(bottom_100_movies, columns=["name", "year", "label"])
            result_df = pd.concat([positive_df, negative_df], ignore_index=True)
            result_df["year"] = result_df["year"].astype(int)
            result_with_original_data = movies.merge(result_df, on=["name", "year"], how="inner")
            movies = result_with_original_data[["id", "name", "year", "label"]]
            self.tables["movies"] = movies.copy()
        elif self.dataset_config.target_schema == 'AdventureWorks2014':
            self.foreign_keys.remove(['SalesOrderHeader', 'SalesPersonID', 'SalesPerson', 'BusinessEntityID'])

    def process(self):
        logging.info(f"Processing dataset: {self.dataset_config.target_schema},"
                     f" Table: {self.dataset_config.target_table}")
        start_time = time.time()
        try:
            db_object = get_database(sql_type=self.dataset_config.sql_type,
                                     database=self.dataset_config.database,
                                     target_schema=self.dataset_config.target_schema,
                                     include_all_schemas=self.dataset_config.include_all_schemas)
            self.tables, self.primary_keys, self.foreign_keys = db_object.get_data()
            self.preprocess_tables()
            self.evaluate(self.tables, self.primary_keys, self.foreign_keys)
        finally:
            logging.info(f"Execution time: {time.time() - start_time:.4f} seconds")

    def propositionalize(self, method, tables, primary_keys, foreign_keys, target_table, target_attribute):
        methods = {
            "wordification": Wordfication,
            "denormalization": Denormalization

        }
        prop_config = PropConfig(tables=tables, primary_keys=primary_keys, foreign_keys=foreign_keys,
                                 target_table=target_table, target_attribute=target_attribute,
                                 max_depth=self.dataset_config.max_depth)

        prop_method = methods.get(method)
        prop_object = prop_method(config=prop_config)
        return prop_object.run()

    @staticmethod
    def encode_labels(labels: pd.Series):
        le = LabelEncoder()
        return le.fit_transform(labels)

    def evaluate(self, tables, primary_keys, foreign_keys):
        for method in self.args.prop_methods:
            features, labels = self.propositionalize(method, tables, primary_keys, foreign_keys,
                                                     target_table=self.dataset_config.target_table,
                                                     target_attribute=self.dataset_config.target_attribute)
            labels = self.encode_labels(labels)
            exp = MLExperiment(feature_config_path=self.args.fe_config,
                               classifier_config_path=self.args.classifier_config, prop_method=method)
            exp.run_experiments(features, labels)
            results = exp.summarize_results(dataset=self.dataset_config.target_schema)
            # Check if the file exists
            file_exists = os.path.isfile(self.args.results_file)

            # Append to the file if it exists, write headers only if the file does not exist
            results.to_csv(self.args.results_file, mode='a', index=False, header=not file_exists)
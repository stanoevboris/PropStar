# import logging
# from itertools import product
# from typing import Dict
#
# import numpy as np
#
# from constants import CLASSIFIER_GRID
# from learning import preprocess_and_split
# from rdm.dataset_processor import DatasetConfig
# from rdm.feature_engineering import get_feature_engineering_callable
# from utils import load_yaml_config, save_results
#
# # TODO this will be deprecated
# class ClassifierProcessor:
#     def __init__(self, classifier_config, dataset_config: DatasetConfig,
#                  args, tables, primary_keys, foreign_keys):
#         self.classifier_config = classifier_config
#         self.dataset_config = dataset_config
#         self.args = args
#         self.lock = threading.Lock()
#         self.initialize_logging()
#
#         self.tables = tables
#         self.primary_keys = primary_keys
#         self.foreign_keys = foreign_keys
#
#
#     def generate_classifier_params(self):
#         config = load_yaml_config(self.classifier_config)
#
#         for classifier in config['classifiers']:
#             for param_set in classifier['params']:
#                 for combination in self.generate_classifier_combinations(param_set):
#                     yield classifier['name'], combination
#
#     @staticmethod
#     def generate_classifier_combinations(params):
#         """
#         Generate all combinations of parameter values.
#
#         Parameters:
#         - params: A dictionary where keys are parameter names and values are lists of parameter values.
#
#         Returns:
#         - A list of dictionaries, each representing a unique combination of parameter values.
#         """
#         keys = params.keys()
#         values_combinations = list(product(*params.values()))
#         return [dict(zip(keys, values)) for values in values_combinations]
#
#     @staticmethod
#     def calculate_stats(metric_name, scores):
#         stats = {
#             f'min_score_{metric_name}': min(scores),
#             f'max_score_{metric_name}': max(scores),
#             f'mean_score_{metric_name}': np.mean(scores),
#             f'std_score_{metric_name}': np.std(scores),
#         }
#         return stats
#
#     def evaluate_all_classifiers(self, target_schema):
#         """Evaluate and classify the dataset using specified classifiers."""
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             # Create a future for each classifier process
#             futures = [
#                 executor.submit(self.evaluate_classifier, classifier_name, classifier_params, )
#                 for classifier_name, classifier_params in self.generate_classifier_params()]
#
#             for future in as_completed(futures):
#                 classifier_name, grid_dict, execution_time = future.result()
#                 logging.info(f"Evaluation of {grid_dict} - {self.dataset_config.target_attribute} - "
#                              f"completed in {execution_time} seconds.")
#
#     def evaluate_classifier(self, classifier_name, classifier_params):
#         """Process a single classifier."""
#         classifier_start_time = time.time()
#         grid_dict = CLASSIFIER_GRID[classifier_name].copy()
#         grid_dict.update(classifier_params)
#         grid_dict['dataset'] = self.dataset_config.target_schema
#         # CLASSIFIER_GRID[classifier_name].update(classifier_params)
#         # CLASSIFIER_GRID[classifier_name]['dataset'] = target_schema
#
#         logging.info("Evaluation of {} - {}".format(
#             grid_dict, self.dataset_config.target_attribute))
#         scores = self.evaluate_classifier_across_folds(classifier_name, grid_dict)
#
#         if scores:
#             with self.lock:
#                 save_results(args=args, scores=scores,
#                              grid_dict=grid_dict)
#
#         classifier_end_time = time.time()
#         classifier_execution_time = classifier_end_time - classifier_start_time
#         # CLASSIFIER_GRID[classifier_name]['execution_time'] = classifier_execution_time
#         return classifier_name, grid_dict, classifier_execution_time
#
#     def evaluate_classifier_across_folds(self, classifier_name, grid_dict):
#         """Evaluate each fold for the dataset."""
#         accuracy_scores, f1_scores, roc_auc_scores, custom_roc_auc_scores = [], [], [], []
#
#         # split_gen = preprocess_and_split(X=self.tables[self.dataset_config.target_table],
#         #                                  num_fold=self.args.folds,
#         #                                  target_attribute=self.dataset_config.target_attribute)
#         #
#         #
#         # with ProcessPoolExecutor(max_workers=1) as executor:
#         #     futures = [executor.submit(self.evaluate_fold, classifier_name, train_index, test_index, grid_dict)
#         #                for train_index, test_index in split_gen]
#         #
#         #     for future in as_completed(futures):
#         #         try:
#         #             acc, f1, auc_roc, custom_roc_auc = future.result()
#         #             accuracy_scores.append(acc)
#         #             f1_scores.append(f1)
#         #             roc_auc_scores.append(auc_roc)
#         #             custom_roc_auc_scores.append(custom_roc_auc)
#         #         except Exception as es:
#         #             logging.error(f"Error in fold evaluation: {es}")
#         #             return None
#         #
#         # return {'acc': accuracy_scores,
#         #         'f1': f1_scores,
#         #         'roc_auc': roc_auc_scores,
#         #         'custom_roc_auc': custom_roc_auc_scores}
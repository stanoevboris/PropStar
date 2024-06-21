import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class MLExperiment:
    def __init__(self, feature_config_path: str, classifier_config_path: str, prop_method: str):
        self.prop_method = prop_method
        self.feature_config = self.load_config(feature_config_path)
        self.classifier_config = self.load_config(classifier_config_path)
        self.results = {}

    @staticmethod
    def load_config(path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def create_pipeline(feature_steps, classifier_info):
        steps = []
        for step in feature_steps:
            module_path, class_name = step['name'].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            transformer_class = getattr(module, class_name)

            transformer_params = step.get('params', {})
            transformer = transformer_class(**transformer_params)

            steps.append((class_name.lower(), transformer))

        class_path, class_name = classifier_info['class'].rsplit('.', 1)
        class_module = __import__(class_path, fromlist=[class_name])
        classifier = getattr(class_module, class_name)()

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score),
            'roc_auc': make_scorer(roc_auc_score, needs_threshold=True)
        }

        param_grid = classifier_info['param_grid']
        search_spaces = {}
        for param, values in param_grid.items():
            if isinstance(values[0], int):
                search_spaces[param] = Integer(min(values), max(values))
            elif isinstance(values[0], float):
                search_spaces[param] = Real(min(values), max(values), prior='log-uniform')
            elif isinstance(values[0], str):
                search_spaces[param] = Categorical(values)

        stratified_cv = StratifiedKFold(n_splits=10)
        bayes_search = BayesSearchCV(classifier, search_spaces=search_spaces, n_iter=10, cv=stratified_cv,
                                     scoring=scoring, refit="roc_auc", verbose=10, n_jobs=3)

        steps.append(('classifier', bayes_search))
        return ImbPipeline(steps)

    def run_experiments(self, X, y):
        for pipeline_name, feature_steps in self.feature_config[self.prop_method].items():
            for classifier_name, classifier_info in self.classifier_config['classifiers'].items():
                print(f"Creating and evaluating pipeline for {pipeline_name} with {classifier_name}")
                pipeline = self.create_pipeline(feature_steps, classifier_info)
                pipeline.fit(X, y)
                self.results[classifier_name, pipeline_name] = pipeline.named_steps['classifier']

    def summarize_results(self, dataset: str) -> pd.DataFrame:
        """
        Summarize the results of a GridSearchCV object when refit is set to False.
        """
        classifiers_summaries = []
        for key, scores in self.results.items():
            classifier_name, pipeline_name = key[0], key[1]
            results_df = pd.DataFrame(scores.cv_results_)

            # Extract parameter and scoring keys
            param_keys = [col for col in results_df.columns if col.startswith('param_')]
            scoring_keys = [col for col in results_df.columns if
                            col.startswith('mean_test_') or col.startswith('std_test_') or col.startswith('rank_test_')]

            important_columns = param_keys + scoring_keys
            current_summary = results_df[important_columns].copy()
            current_summary['classifier'] = classifier_name
            current_summary['feature_engineering_type'] = pipeline_name
            current_summary['dataset'] = dataset

            # Sort by the first scoring metric rank (you can adjust this as needed)
            first_rank_col = 'rank_test_roc_auc'
            current_summary_sorted = current_summary.sort_values(by=first_rank_col)
            classifiers_summaries.append(current_summary_sorted)

        return pd.concat(classifiers_summaries, ignore_index=True)
# Usage:
# exp = MLExperiment('feature_engineering.yaml', 'classifiers.yaml')
# experiment_results = exp.run_experiments()

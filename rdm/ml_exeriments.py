from typing import Dict, Callable
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.pipeline import Pipeline as ImbPipeline


from rdm.constants import scoring_metrics
from rdm.utils import load_yaml_config


# TODO: include the cross-validation folds in this class
class MLExperiment:
    def __init__(self, feature_config_path: str, classifier_config_path: str, prop_method: str, problem_type: str):
        self.prop_method = prop_method
        self.problem_type = problem_type
        self.feature_config = load_yaml_config(feature_config_path)
        self.classifier_config = load_yaml_config(classifier_config_path)

        self.scoring_metrics = self.init_scoring_metrics()
        self.refit_metric = self.get_refit_metric()
        self.results = {}

    def init_scoring_metrics(self) -> Dict[str, Callable]:
        return scoring_metrics.get(self.problem_type)

    def get_refit_metric(self):
        if self.problem_type == 'binary_classification':
            return 'roc_auc'
        elif self.problem_type == 'multiclass_classification':
            return 'f1_macro'  # or 'accuracy' or 'roc_auc_ovr'
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    @staticmethod
    def create_pipeline(feature_steps, classifier_info, scoring, refit_metric):
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

        steps.append(('classifier', classifier))

        pipeline = ImbPipeline(steps)

        param_grid = classifier_info['param_grid']
        search_spaces = {}
        for param, values in param_grid.items():
            param_name = f'classifier__{param}'
            if isinstance(values[0], int):
                search_spaces[param_name] = Integer(min(values), max(values))
            elif isinstance(values[0], float):
                search_spaces[param_name] = Real(min(values), max(values), prior='log-uniform')
            elif isinstance(values[0], str):
                search_spaces[param_name] = Categorical(values)

        stratified_cv = StratifiedKFold(n_splits=10)
        bayes_search = BayesSearchCV(pipeline, search_spaces=search_spaces, n_iter=5, cv=stratified_cv,
                                     scoring=scoring, refit=refit_metric, verbose=10, n_jobs=1)

        return bayes_search

    def run_experiments(self, X, y):
        for pipeline_name, feature_steps in self.feature_config[self.prop_method].items():
            for classifier_name, classifier_info in self.classifier_config['classifiers'].items():
                print(f"Creating and evaluating pipeline for {pipeline_name} with {classifier_name}")
                pipeline = self.create_pipeline(feature_steps, classifier_info, self.scoring_metrics, self.refit_metric)
                pipeline.fit(X, y)
                self.results[classifier_name, pipeline_name] = pipeline

    def summarize_results(self, dataset: str) -> pd.DataFrame:
        """
        Summarize the results of a GridSearchCV object when refit is set to False.
        """
        classifiers_summaries = []
        for key, bayes_search in self.results.items():
            classifier_name, pipeline_name = key
            results_df = pd.DataFrame(bayes_search.cv_results_)

            # Extract parameter and scoring keys
            param_keys = [col for col in results_df.columns if col.startswith('param_')]
            scoring_keys = [col for col in results_df.columns if
                            col.startswith('mean_test_') or col.startswith('std_test_') or col.startswith(
                                'rank_test_')]

            important_columns = param_keys + scoring_keys
            current_summary = results_df[important_columns].copy()
            current_summary['classifier'] = classifier_name
            current_summary['feature_engineering_type'] = pipeline_name
            current_summary['dataset'] = dataset
            current_summary['problem_type'] = self.problem_type

            # Sort by the first scoring metric rank (you can adjust this as needed)
            first_rank_col = f'rank_test_{self.refit_metric}'
            current_summary_sorted = current_summary.sort_values(by=first_rank_col)
            classifiers_summaries.append(current_summary_sorted)

        return pd.concat(classifiers_summaries, ignore_index=True)
# Usage:
# exp = MLExperiment('feature_engineering.yaml', 'classifiers.yaml')
# experiment_results = exp.run_experiments()

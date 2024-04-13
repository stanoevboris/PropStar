import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from sklearn.base import is_classifier
from sklearn.utils import all_estimators


class EstimatorSelectionHelper:
    def __init__(self, search_space, cv=None):
        self.search_space = search_space
        self.models = list(search_space.keys())
        self.grid_searches = {}
        self.cv = cv or StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        self.callable_dict = {name: cls for name, cls in all_estimators() if is_classifier(cls)}

    def fit(self, X, y, fe_pipeline=None, n_jobs=3, verbose=1, scoring=None, refit=False):
        for model_name in self.models:
            if model_name not in self.callable_dict:
                raise ValueError(f"{model_name} is not a recognized sklearn classifier")
            print(f"Running GridSearchCV for {model_name}.")
            classifier = self.callable_dict[model_name]()
            smote_enn = SMOTEENN(smote=SMOTE(random_state=42),
                                 enn=EditedNearestNeighbours(sampling_strategy='majority'))
            pipeline = ImbPipeline([
                ('feature_engineering', fe_pipeline) if fe_pipeline else ('passthrough', 'passthrough'),
                ('sampler', smote_enn),
                ('classifier', classifier)
            ])
            params = self.search_space[model_name]
            gs = GridSearchCV(pipeline, params, cv=self.cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit, return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[model_name] = gs

    def summary(self):
        results = []
        for model_name, grid_search in self.grid_searches.items():
            for i in range(len(grid_search.cv_results_['params'])):
                result = {
                    'model': model_name,
                    'params': grid_search.cv_results_['params'][i],
                    'mean_test_accuracy': grid_search.cv_results_.get('mean_test_accuracy', [None])[i],
                    'std_test_accuracy': grid_search.cv_results_.get('std_test_accuracy', [None])[i],
                    'mean_test_f1': grid_search.cv_results_.get('mean_test_f1', [None])[i],
                    'std_test_f1': grid_search.cv_results_.get('std_test_f1', [None])[i],
                    'mean_test_roc_auc': grid_search.cv_results_.get('mean_test_roc_auc_score', [None])[i],
                    'std_test_roc_auc': grid_search.cv_results_.get('std_test_roc_auc_score', [None])[i],
                }
                results.append(result)
        summary_df = pd.DataFrame(results)
        return summary_df

# Example usage of this class might require defining a proper search_space before initializing an object.

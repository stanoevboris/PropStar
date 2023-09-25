import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by=None):
        def row(key, acc_scores, roc_auc_scores, params):
            d = {
                'estimator': key,
                'min_score_acc': min(acc_scores),
                'max_score_acc': max(acc_scores),
                'mean_score_acc': np.mean(acc_scores),
                'std_score_acc': np.std(acc_scores),
                'min_score_roc_auc': min(roc_auc_scores),
                'max_score_roc_auc': max(roc_auc_scores),
                'mean_score_roc_auc': np.mean(roc_auc_scores),
                'std_score_roc_auc': np.std(roc_auc_scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            acc_scores = []
            roc_auc_scores = []
            for i in range(self.grid_searches[k].cv):
                acc_key = "split{}_test_accuracy".format(i)
                acc_result = self.grid_searches[k].cv_results_[acc_key]
                acc_scores.append(acc_result.reshape(len(params), 1))

                roc_auc_key = "split{}_test_roc_auc".format(i)
                roc_auc_result = self.grid_searches[k].cv_results_[roc_auc_key]
                roc_auc_scores.append(roc_auc_result.reshape(len(params), 1))

            all_acc_scores = np.hstack(acc_scores)
            all_roc_auc_scores = np.hstack(roc_auc_scores)
            for p, acc_scores, roc_auc_scores in zip(params, all_acc_scores, all_roc_auc_scores):
                rows.append((row(k, acc_scores, roc_auc_scores, p)))

        df = pd.concat(rows, axis=1).T.sort_values(sort_by, ascending=False)

        columns = [c for c in df.columns]

        return df[columns]

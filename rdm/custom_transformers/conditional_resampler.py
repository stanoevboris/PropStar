import importlib
from collections import Counter
from sklearn.base import BaseEstimator


class ConditionalResampler(BaseEstimator):
    def __init__(self, resampler_class_name='imblearn.over_sampling.SMOTE', imbalance_threshold=0.1,
                 resampler_params=None):
        if resampler_params is None:
            resampler_params = {'sampling_strategy': 'auto'}
        self.resampler_class_name = resampler_class_name
        self.imbalance_threshold = imbalance_threshold
        self.resampler_params = resampler_params
        self.resampler = self._get_resampler_instance()
        self.need_resample = None

    def _get_resampler_instance(self):
        module_name, class_name = self.resampler_class_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        resampler_class = getattr(module, class_name)
        return resampler_class(**self.resampler_params)

    def fit_resample(self, X, y):
        class_counts = Counter(y)
        total_count = sum(class_counts.values())
        majority_class_count = max(class_counts.values())

        imbalance_ratios = {cls: count / majority_class_count for cls, count in class_counts.items()}
        minority_class_ratio = min(imbalance_ratios.values())

        if minority_class_ratio > self.imbalance_threshold:
            # Classes are not significantly imbalanced
            self.need_resample = False
            return X, y
        else:
            # Apply the resampler
            self.need_resample = True
            return self.resampler.fit_resample(X, y)

    def _fit_resample(self, X, y):
        return self.fit_resample(X, y) if self.need_resample else (X, y)

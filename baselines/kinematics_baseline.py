import sys
sys.path.append('C:\Projects\Private\PropStar')
sys.path.append('C:\Projects\Private\PropStar\datasets')
from gridsearch.EstimatorSelectionHelper import EstimatorSelectionHelper
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score

search_space = {
    'RandomForestClassifier': {
        'classifier__n_estimators': [16, 32],
        'classifier__max_depth': [None],  # 'null' is equivalent to None in Python
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
    },
    'ExtraTreesClassifier': {
        'classifier__n_estimators': [16, 32],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
    },
    'AdaBoostClassifier': {
        'classifier__n_estimators': [16, 32],
        'classifier__learning_rate': [0.01],
    },
    'GradientBoostingClassifier': {
        'classifier__n_estimators': [16, 32],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [5],
    }
}

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from scipy.fftpack import fft


# Custom transformer for time-based features
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column_name='timestamp'):
        self.time_column_name = time_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Ensure that the input is a pandas DataFrame with datetime dtype for the timestamp column
        time_data = pd.to_datetime(X, errors='coerce')
        if time_data.dtype == '<M8[ns]':  # Check if it's a datetime Series
            # Extract time-related features
            features = pd.DataFrame({
                'year': time_data.dt.year,
                'month': time_data.dt.month,
                'day': time_data.dt.day,
                'hour': time_data.dt.hour,
                'minute': time_data.dt.minute,
                'second': time_data.dt.second,

                # 'sin_hour': np.sin(2 * np.pi * time_data.dt.hour / 24),
                # 'cos_hour': np.cos(2 * np.pi * time_data.dt.hour / 24)
            })
            return features
        else:
            raise ValueError("Input must be a pandas Series with datetime64 dtype.")


# Custom transformer for statistical features
class StatisticalFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=54):  # Example window size ~10 seconds
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        rolling = X.rolling(window=self.window_size, min_periods=1)
        return np.hstack([rolling.mean().values, rolling.std().values])


# Custom transformer for frequency domain features
class FFTFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.abs(fft(X, axis=0))


# Column selector helper
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
timestamp_column = ['timestamp']

# Define the feature engineering pipeline
feature_engineering_pipeline = FeatureUnion([
    ('time_features', ColumnTransformer([('time_extractor', TimeFeaturesExtractor(), 'timestamp')])),
    ('sensor_statistical', Pipeline([
        ('selector', ColumnSelector(sensor_columns)),
        ('statistical_features', StatisticalFeaturesExtractor()),
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values if necessary
        ('scaler', StandardScaler())  # Scaling features
    ])),
    ('sensor_fft', Pipeline([
        ('selector', ColumnSelector(sensor_columns)),
        ('fft', FFTFeaturesExtractor()),
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values if necessary
        ('scaler', StandardScaler())  # Scaling FFT features
    ])),

    ('wrist_encoder', Pipeline([
        ('selector', ColumnSelector(['wrist'])),
        ('encoder', OneHotEncoder())
    ]))
])

# Combine feature engineering pipeline with a model in a full pipeline
full_pipeline = Pipeline([
    ('features', feature_engineering_pipeline),
    # Add your classifier pipeline here
])

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc_score': make_scorer(roc_auc_score)
}

data = pd.read_csv('../datasets/Kinematics_Data.csv')
data.drop('username', axis=1, inplace=True)
data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y-%m-%d %H:%M:%S:%f')
data.drop(['date', 'time'], axis=1, inplace=True)

X = data.drop('activity', axis=1)
y = data['activity']
es = EstimatorSelectionHelper(search_space=search_space)
es.fit(X, y, fe_pipeline=feature_engineering_pipeline, scoring=scoring, n_jobs=5, verbose=10)
es.summary()
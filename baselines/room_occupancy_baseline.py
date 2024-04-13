import sys

sys.path.append('C:\Projects\Private\PropStar')
sys.path.append('C:\Projects\Private\PropStar\datasets')
from gridsearch.EstimatorSelectionHelper import EstimatorSelectionHelper
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
import datetime as dt

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
    },
    'XGBClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [6],
        'classifier__subsample': [0.8],
        'classifier__colsample_bytree': [0.8]
    },
    # 'LGBMClassifier': {
    #     'classifier__n_estimators': [100, 200],
    #     'classifier__learning_rate': [0.01, 0.1],
    #     'classifier__max_depth': [-1],  # -1 for no limit
    #     'classifier__num_leaves': [31],
    #     'classifier__subsample': [0.8],
    #     'classifier__colsample_bytree': [0.8]
    # }
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
    def __init__(self, window_size=60):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # .set_index('timestamp')
        rolling = X.rolling(window=60, min_periods=1)
        return np.hstack([rolling.mean().values, rolling.std().values])


# Column selector helper
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


sensor_columns = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                  'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

feature_engineering_pipeline = ColumnTransformer([
    ('datetime', TimeFeaturesExtractor(), 'timestamp'),
    ('sensors', Pipeline([
        ('rolling', StatisticalFeaturesExtractor()),
        ('scale', StandardScaler()),
        ('imputer', SimpleImputer())
    ]), sensor_columns),
    ('categorical', OneHotEncoder(), ['S6_PIR', 'S7_PIR'])  # Assuming PIR sensors are categorical
])

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc_score': make_scorer(roc_auc_score)
}
from ucimlrepo import fetch_ucirepo

# fetch dataset
room_occupancy_estimation = fetch_ucirepo(id=864)

# data (as pandas dataframes)
X = room_occupancy_estimation.data.features
y = room_occupancy_estimation.data.targets

y.loc[y['Room_Occupancy_Count'] > 0, 'Room_Occupancy_Count'] = 1
y = y.to_numpy()
X['timestamp'] = pd.to_datetime(X['Date'] + ' ' + X['Time'])
X.drop(['Date', 'Time'], axis=1, inplace=True)

es = EstimatorSelectionHelper(search_space=search_space)
es.fit(X, y, fe_pipeline=feature_engineering_pipeline, scoring=scoring, n_jobs=5, verbose=10)
results = es.summary()

results.to_csv("room_occupancy.csv", mode='w')

import os
import sys

# sys.path is a list of absolute path strings
sys.path.append('C:\Projects\Private\PropStar')
sys.path.append('C:\Projects\Private\PropStar\datasets')
from gridsearch.EstimatorSelectionHelper import EstimatorSelectionHelper
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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
    'LGBMClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [-1],  # -1 for no limit
        'classifier__num_leaves': [31],
        'classifier__subsample': [0.8],
        'classifier__colsample_bytree': [0.8]
    }
}

dataset_group = pd.read_csv('../datasets/indoor_user_movement_rss_data/groups/MovementAAL_DatasetGroup.csv')
dataset_group['sequence_ID'] = dataset_group['#sequence_ID']
dataset_group['dataset_ID'] = dataset_group[' dataset_ID']
dataset_group.drop(['#sequence_ID', ' dataset_ID'], axis=1, inplace=True)

paths = pd.read_csv('../datasets/indoor_user_movement_rss_data/groups/MovementAAL_Paths.csv')
paths['sequence_ID'] = paths['#sequence_ID']
paths['path_ID'] = paths[' path_ID']
paths.drop(['#sequence_ID', ' path_ID'], axis=1, inplace=True)

target = pd.read_csv('../datasets/indoor_user_movement_rss_data/MovementAAL_target.csv')
target['class_label'] = target[' class_label']
target['sequence_ID'] = target['#sequence_ID']
target.drop(['#sequence_ID', ' class_label'], axis=1, inplace=True)

files = os.listdir('../datasets/indoor_user_movement_rss_data/dataset')
movements_df = pd.DataFrame()
for file in files:
    if file.startswith('.'):
        continue
    seq_id = file.split("_")[2].replace(".csv", "")
    file_df = pd.read_csv(f"../datasets/indoor_user_movement_rss_data/dataset/{file}")
    file_df['seq_id'] = seq_id
    movements_df = pd.concat([movements_df, file_df])

movements_df.reset_index(inplace=True)
movements_df.drop("index", axis=1, inplace=True)
movements_df.reset_index(inplace=True, names='id')
movements_df['RSS_anchor1'] = movements_df['#RSS_anchor1']
movements_df.drop('#RSS_anchor1', axis=1, inplace=True)
movements_df['seq_id'] = pd.to_numeric(movements_df['seq_id'])

denormalized_table = target \
    .merge(movements_df, how='left', left_on=['sequence_ID'], right_on=['seq_id']) \
    .merge(dataset_group, how='left', on=['sequence_ID']) \
    .merge(paths, how='left', on=['sequence_ID']) \
    .rename(columns=lambda x: x.strip()) \
    .drop(['seq_id', 'sequence_ID', 'id'], axis=1)

numerical_cols = ['RSS_anchor1', 'RSS_anchor2', 'RSS_anchor3', 'RSS_anchor4']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Example: handle missing values
    ('scaler', StandardScaler())  # Scale features
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder())  # Convert categorical data
])

# Combine into a single ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
])

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc_score': make_scorer(roc_auc_score)
}
X = denormalized_table.drop('class_label', axis=1)
y = denormalized_table['class_label']
y = (y + 1) // 2
es = EstimatorSelectionHelper(search_space=search_space)
es.fit(X, y, fe_pipeline=preprocessor, scoring=scoring, n_jobs=5, verbose=10)
es.summary().to_csv('indoor_user_movement.csv', mode='w')

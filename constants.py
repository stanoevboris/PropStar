from classification import traditional_learner_tfidf_classification, traditional_learner_woe_classification, \
    prop_drm_tfidf_classification, prop_drm_woe_classification, prop_star_tfidf_classification, \
    prop_star_woe_classification
# from propositionalization import generate_custom_relational_words, generate_relational_words

# FEATURE_FUNC = {
#     "woe": generate_custom_relational_words,
#     "sklearn_tfidf": generate_relational_words
# }

COMMON_CLASSIFIERS = {
    'sklearn_tfidf': traditional_learner_tfidf_classification,
    'woe': traditional_learner_woe_classification
}

CLASSIFIER_FUNC = {
    "DRM": {
        'sklearn_tfidf': prop_drm_tfidf_classification,
        'woe': prop_drm_woe_classification
    },
    "starspace": {
        'sklearn_tfidf': prop_star_tfidf_classification,
        'woe': prop_star_woe_classification
    },
    "svm_learner": COMMON_CLASSIFIERS,
    "extra_tree_learner": COMMON_CLASSIFIERS,
    "random_forest_learner": COMMON_CLASSIFIERS,
    "ada_boost_learner": COMMON_CLASSIFIERS,
    "gradient_boost_learner": COMMON_CLASSIFIERS,
    "xgboost_learner": COMMON_CLASSIFIERS,
    "lightgbm_learner": COMMON_CLASSIFIERS,
    "catboost_learner": COMMON_CLASSIFIERS
}

COMMON_PARAMS = {
    'epochs': 5,
    'representation_type': "woe",
    'dataset': None  # Assuming you'll set this as needed
}

CLASSIFIER_GRID = {
    "DRM": {
        **COMMON_PARAMS, 'learner': 'DRM', 'learning_rate': 0.001, 'hidden_size': 16,
        'dropout': 0.1, 'num_features': 30000
    },
    "starspace": {
        **COMMON_PARAMS, 'learner': 'starspace', 'learning_rate': 0.001, 'hidden_size': 16,
        'negative_samples_limit': 5, 'negative_search_limit': 10, 'num_features': 30000
    },
    "svm_learner": {
        **COMMON_PARAMS, 'learner': 'svm_learner', 'kernel': 'linear', 'C': 1, 'gamma': 'scale'
    },
    "extra_tree_learner": {
        **COMMON_PARAMS, 'learner': 'extra_tree_learner', 'n_estimators': 16, 'max_depth': None,
        'min_samples_split': 2, 'min_samples_leaf': 2
    },
    "random_forest_learner": {
        **COMMON_PARAMS, 'learner': 'random_forest_learner', 'n_estimators': 16, 'max_depth': None,
        'min_samples_split': 2, 'min_samples_leaf': 2
    },
    "ada_boost_learner": {
        **COMMON_PARAMS, 'learner': 'ada_boost_learner', 'n_estimators': 16, 'learning_rate': 0.001
    },
    "gradient_boost_learner": {
        **COMMON_PARAMS, 'learner': 'gradient_boost_learner', 'n_estimators': 16, 'learning_rate': 0.001,
        'max_depth': None
    },
    "xgboost_learner": {
        **COMMON_PARAMS, 'learner': 'xgboost_learner', 'n_estimators': 16, 'learning_rate': 0.001, 'max_depth': None,
        'subsample': 0.8, 'colsample_bytree': 0.8
    },
    "lightgbm_learner": {
        **COMMON_PARAMS, 'learner': 'lightgbm_learner', 'n_estimators': 16, 'learning_rate': 0.001, 'max_depth': None,
        'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8
    },
    "catboost_learner": {
        **COMMON_PARAMS, 'learner': 'catboost_learner', 'iterations': 16, 'learning_rate': 0.001,
        'depth': 6, 'l2_leaf_reg': 1
    }
}

import json
import argparse

from collections import Counter

import yaml

from propositionalization import (get_data,
                                  generate_relational_words,
                                  generate_custom_relational_words)
from classification import *
from utils import save_results, preprocess_tables

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

feature_func = {
    "woe": generate_custom_relational_words,
    "sklearn_tfidf": generate_relational_words
}

common_classifiers = {
    'sklearn_tfidf': traditional_learner_tfidf_classification,
    'woe': traditional_learner_woe_classification
}

classifier_func = {
    "DRM": {
        'sklearn_tfidf': prop_drm_tfidf_classification,
        'woe': prop_drm_woe_classification
    },
    "starspace": {
        'sklearn_tfidf': prop_star_tfidf_classification,
        'woe': prop_star_woe_classification
    },
    "svm_learner": common_classifiers,
    "extra_tree_learner": common_classifiers,
    "random_forest_learner": common_classifiers,
    "ada_boost_learner": common_classifiers,
    "gradient_boost_learner": common_classifiers,
    "xgboost_learner": common_classifiers,
    "lightgbm_learner": common_classifiers,
    "catboost_learner": common_classifiers
}


def log_dataset_info(train_features, test_features):
    """
    Logs information about the dataset, working with both DataFrames and sparse matrices.
    """
    num_train_records, num_train_features = train_features.shape
    num_test_records, num_test_features = test_features.shape

    total_records = num_train_records + num_test_records

    logging.info(f"Dataset number of records: {total_records}")
    logging.info(f"Train set number of features: {num_train_features}")
    logging.info(f"Test set number of features: {num_test_features}")


def calculate_positive_class_percentage(train_classes, test_classes, representation_type):
    """
    Calculate the percentage occurrence of the positive class in the combined class list or array,
    handling both DataFrames and numpy arrays. The positive class is defined by the 'positive_label'.

    Parameters:
    - train_classes: Training class labels.
    - test_classes: Test class labels.
    - representation_type: Type of representation, affecting how data is processed.

    Returns:
    - The percentage of occurrences of the positive class.
    """
    if representation_type == 'woe':
        # Handling DataFrame: Extracting values as a list
        all_values = list(train_classes.values()) + list(test_classes.values())
    else:
        # Handling numpy array (for sparse matrix, ensure conversion to numpy array beforehand)
        all_values = np.concatenate((train_classes, test_classes))

    # Count occurrences of each class
    occurrences = Counter(all_values)

    # Calculate total number of occurrences
    total_count = sum(occurrences.values())

    # Calculate the percentage of the positive class
    positive_class_count = occurrences.get(1, 0)
    positive_class_percentage = (positive_class_count / total_count) * 100 if total_count > 0 else 0

    return positive_class_percentage


def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="results_file.csv", help="The path to the results file")
    parser.add_argument("--learner", default="random_forest_learner")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--num_features", default=30000, type=int, help="Number of features")
    parser.add_argument("--hidden_size", default=16, type=int, help="Embedding dimension")
    parser.add_argument("--negative_samples_limit", default=5, type=int, help="Max number of negative samples")
    parser.add_argument("--negative_search_limit", default=10, type=int, help="Negative search limit")
    parser.add_argument("--n_estimators", default=16, type=int, help="No. of estimators for traditional classifiers")
    parser.add_argument("--kernel", default='linear', type=str, help="Kernel for SVC")
    parser.add_argument("--C", default=1, type=int, help="Regularization parameter for SVC")
    parser.add_argument("--gamma", default='scale', help="Kernel coefficient for SVC")
    parser.add_argument("--depth", default=6, type=int, help="Depth param for CatBoost classifier")
    parser.add_argument("--max_depth", default=None, type=int_or_none, help="Max Depth param for various classifiers")
    parser.add_argument("--min_samples_split", default=2, type=int, help="Min samples split")
    parser.add_argument("--min_samples_leaf", default=2, type=int, help="Min samples leaf")
    parser.add_argument("--subsample", default=0.8, type=float, help="Subsample param")
    parser.add_argument("--colsample_bytree", default=0.8, type=float,
                        help="Colsample by tree XGBoost LightGBM LightGBM")
    parser.add_argument("--num_leaves", default=31, type=int, help="Number of leaves")
    parser.add_argument("--l2_leaf_reg", default=1, type=int, help="L2 for Leaf Regularization")
    parser.add_argument("--representation_type", default="woe", type=str, help="Type of representation")
    return parser.parse_args()


def build_grid_dict(args):
    shared_params = {'learner': args.learner, 'epochs': args.epochs, 'folds': args.folds,
                     'representation_type': args.representation_type, 'dataset': 'none'}
    # Default parameters for all learners, with specific overrides as needed
    grid = {
        "DRM": {**shared_params, 'learning_rate': args.learning_rate, 'hidden_size': args.hidden_size,
                'dropout': args.dropout, 'num_features': args.num_features},
        "starspace": {**shared_params, 'learning_rate': args.learning_rate, 'hidden_size': args.hidden_size,
                      'negative_samples_limit': args.negative_samples_limit,
                      'negative_search_limit': args.negative_search_limit,
                      'num_features': args.num_features},
        "svm_learner": {**shared_params, 'kernel': args.kernel, 'C': args.C, 'gamma': args.gamma},
        "extra_tree_learner": {**shared_params, 'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                               'min_samples_split': args.min_samples_split, 'min_samples_leaf': args.min_samples_leaf},
        "random_forest_learner": {**shared_params, 'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                                  'min_samples_split': args.min_samples_split,
                                  'min_samples_leaf': args.min_samples_leaf},
        "ada_boost_learner": {**shared_params, 'n_estimators': args.n_estimators, 'learning_rate': args.learning_rate},
        "gradient_boost_learner": {**shared_params, 'n_estimators': args.n_estimators,
                                   'learning_rate': args.learning_rate, 'max_depth': args.max_depth},
        "xgboost_learner": {**shared_params, 'n_estimators': args.n_estimators, 'learning_rate': args.learning_rate,
                            'max_depth': args.max_depth, 'subsample': args.subsample,
                            'colsample_bytree': args.colsample_bytree},
        "lightgbm_learner": {**shared_params, 'n_estimators': args.n_estimators, 'learning_rate': args.learning_rate,
                             'max_depth': args.max_depth, 'num_leaves': args.num_leaves, 'subsample': args.subsample,
                             'colsample_bytree': args.colsample_bytree},
        "catboost_learner": {**shared_params, 'iterations': args.n_estimators, 'learning_rate': args.learning_rate,
                             'depth': args.depth, 'l2_leaf_reg': args.l2_leaf_reg}
    }
    return grid


def process_dataset(dataset, args, grid_dict):
    """Process a single dataset."""
    start_time = time.time()
    sql_type, database = dataset['sql_type'], dataset['database']
    target_schema = dataset['target_schema']
    target_table = dataset['target_table']
    target_attribute = dataset['target_column']
    include_all_schemas = dataset['include_all_schemas']

    logging.info(
        f"Running for dataset: {target_schema}, target_table: {target_table}, target_attribute {target_attribute}")
    args_dict = vars(args)
    args_dict['dataset'] = target_schema
    grid_dict[args.learner]['dataset'] = target_schema

    tables, primary_keys, fkg = get_data(sql_type=sql_type,
                                         database=database,
                                         target_schema=target_schema,
                                         include_all_schemas=include_all_schemas)
    tables = preprocess_tables(target_schema=target_schema, tables=tables)

    scores = transform_and_classify(tables, target_table, target_attribute, args, grid_dict, primary_keys, fkg)

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution time: {execution_time:.4f} seconds")
    grid_dict[args.learner]['total_execution_time'] = f"{execution_time:.4f}"

    save_results(args=args, dataset=target_schema, scores=scores,
                 grid_dict=grid_dict[args.learner])


def transform_and_classify(tables, target_table, target_attribute, args, grid_dict, primary_keys, fkg):
    """Evaluate and classify, returning scores."""
    split_gen = preprocess_and_split(X=tables[target_table], num_fold=args.folds, target_attribute=target_attribute)
    accuracy_scores, f1_scores, roc_auc_scores, custom_roc_auc_scores = [], [], [], []
    logging.info("Evaluation of {} - {}".format(
        grid_dict[args.learner], target_attribute))
    for train_index, test_index in split_gen:
        generate_relational_words_func = feature_func[args.representation_type]
        train_features, train_classes, encoder = generate_relational_words_func(
            tables,
            fkg,
            target_table,
            target_attribute,
            relation_order=(1, 1),
            indices=train_index,
            vectorization_type=args.representation_type,
            num_features=args.num_features,
            primary_keys=primary_keys)
        test_features, test_classes = generate_relational_words_func(
            tables,
            fkg,
            target_table,
            target_attribute,
            relation_order=(1, 1),
            encoder=encoder,
            indices=test_index,
            vectorization_type=args.representation_type,
            num_features=args.num_features,
            primary_keys=primary_keys)

        log_dataset_info(train_features=train_features, test_features=test_features)

        labels_occurrence_percentage = calculate_positive_class_percentage(
            train_classes, test_classes, args.representation_type
        )
        grid_dict[args.learner]['labels_occurrence_percentage'] = labels_occurrence_percentage
        classify_func = classifier_func[args.learner][args.representation_type]
        try:
            acc, f1, auc_roc, custom_roc_auc = classify_func(args=args,
                                                             train_features=train_features,
                                                             train_classes=train_classes,
                                                             test_features=test_features,
                                                             test_classes=test_classes)
            accuracy_scores.append(acc)
            f1_scores.append(f1)
            roc_auc_scores.append(auc_roc)
            custom_roc_auc_scores.append(custom_roc_auc)
        except Exception as es:
            print(es)
            return None

    scores = {'acc': accuracy_scores,
              'f1': f1_scores,
              'roc_auc': roc_auc_scores,
              'custom_roc_auc': custom_roc_auc_scores}
    return scores


def setup_directory(directory_path):
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory {directory_path} created.")


def main():
    args = parse_arguments()
    grid_dict = build_grid_dict(args)

    setup_directory("tmp")

    try:
        with open('datasets.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("datasets.yaml file not found.")
        return

    for dataset in config['datasets']:
        if not dataset.get('enabled', True):
            continue
        process_dataset(dataset=dataset, args=args, grid_dict=grid_dict)


if __name__ == "__main__":
    main()

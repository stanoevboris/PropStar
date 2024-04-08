import logging
import argparse

import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock

from constants import FEATURE_FUNC, CLASSIFIER_FUNC, CLASSIFIER_GRID
from learning import preprocess_and_split
from propositionalization import get_data
from utils import save_results, preprocess_tables, setup_directory, generate_classifier_params, log_dataset_info, \
    calculate_positive_class_percentage, load_yaml_config

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

save_results_lock = Lock()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="experiments.csv", help="The path to the results file")
    parser.add_argument("--config_file", default="wordification_config.yaml", help="The path to the classifiers config")
    parser.add_argument('--folds', type=int, default=10)
    return parser.parse_args()


def process_dataset(dataset, args):
    """
    Process and evaluate a single dataset by extracting relevant information,
    fetching data based on the dataset configuration, preprocessing tables, and
    initiating the evaluation of the dataset.

    Args:
    - dataset (dict): Information about the dataset including SQL type, database,
      target schema/table/column, and inclusion flag for all schemas.
    - args: Command line arguments or any configuration parameters required for processing.
    """
    start_time = time.time()
    sql_type, database = dataset['sql_type'], dataset['database']
    target_schema = dataset['target_schema']
    target_table = dataset['target_table']
    target_attribute = dataset['target_column']
    include_all_schemas = dataset['include_all_schemas']

    logging.info(
        f"Running for dataset: {target_schema}, target_table: {target_table}, target_attribute {target_attribute}")

    tables, primary_keys, fkg = get_data(sql_type=sql_type,
                                         database=database,
                                         target_schema=target_schema,
                                         include_all_schemas=include_all_schemas)
    tables = preprocess_tables(target_schema=target_schema, tables=tables)

    evaluate_dataset(target_schema, tables, target_table, target_attribute, args, primary_keys, fkg)

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution time: {execution_time:.4f} seconds")


def evaluate_dataset(target_schema, tables, target_table, target_attribute, args, primary_keys, fkg):
    """
    Evaluate and classify the dataset, returning scores for each classifier specified
    in the configuration file.
    """

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Create a future for each classifier process
        futures = [executor.submit(process_classifier, target_schema, classifier_name, classifier_params, tables,
                                   target_table, target_attribute, args, primary_keys, fkg)
                   for classifier_name, classifier_params in generate_classifier_params(args.config_file)]

        for future in as_completed(futures):
            classifier_name, grid_dict, execution_time = future.result()
            logging.info(f"Evaluation of {grid_dict} - {target_attribute} - "
                         f"completed in {execution_time} seconds.")


def process_classifier(target_schema, classifier_name, classifier_params, tables, target_table, target_attribute, args,
                       primary_keys, fkg):
    """
    Process a single classifier.

    This function is intended to be run in parallel for each classifier.
    """
    classifier_start_time = time.time()
    grid_dict = CLASSIFIER_GRID[classifier_name].copy()
    grid_dict.update(classifier_params)
    grid_dict['dataset'] = target_schema
    # CLASSIFIER_GRID[classifier_name].update(classifier_params)
    # CLASSIFIER_GRID[classifier_name]['dataset'] = target_schema

    logging.info("Evaluation of {} - {}".format(
        grid_dict, target_attribute))
    scores = process_folds(classifier_name, tables, target_table, target_attribute, args.folds, primary_keys, fkg,
                           grid_dict)

    if scores:
        with save_results_lock:
            save_results(args=args, scores=scores,
                         grid_dict=grid_dict)

    classifier_end_time = time.time()
    classifier_execution_time = classifier_end_time - classifier_start_time
    # CLASSIFIER_GRID[classifier_name]['execution_time'] = classifier_execution_time
    return classifier_name, grid_dict, classifier_execution_time


def process_folds(classifier_name, tables, target_table, target_attribute, folds, primary_keys, fkg, grid_dict):
    """
        Processes each fold for the dataset evaluation using the specified classifier.

        Args:
        - classifier_name (str): Name of the classifier to use.
        - tables (dict): Preprocessed table data.
        - target_table (str): The target table in the dataset.
        - target_attribute (str): The attribute/column in the table to be classified.
        - folds (int): Number of folds to split the dataset into for cross-validation.
        - primary_keys (dict): Primary keys for the tables.
        - fkg: Foreign key graphs or relationship information.

        Returns:
        - dict: Scores accumulated from evaluating each fold.
        """
    accuracy_scores, f1_scores, roc_auc_scores, custom_roc_auc_scores = [], [], [], []
    split_gen = preprocess_and_split(X=tables[target_table], num_fold=folds, target_attribute=target_attribute)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate_dataset_fold, classifier_name, train_index, test_index,
                                   tables, target_table, target_attribute,
                                   primary_keys, fkg, grid_dict)
                   for train_index, test_index in split_gen]

        for future in as_completed(futures):
            try:
                acc, f1, auc_roc, custom_roc_auc = future.result()
                accuracy_scores.append(acc)
                f1_scores.append(f1)
                roc_auc_scores.append(auc_roc)
                custom_roc_auc_scores.append(custom_roc_auc)
            except Exception as es:
                logging.error(f"Error in fold evaluation: {es}")
                return None

    return {'acc': accuracy_scores,
            'f1': f1_scores,
            'roc_auc': roc_auc_scores,
            'custom_roc_auc': custom_roc_auc_scores}


def evaluate_dataset_fold(classifier_name, train_index, test_index, tables, target_table, target_attribute,
                          primary_keys, fkg, grid_dict):
    generate_relational_words_func = FEATURE_FUNC[grid_dict.get('representation_type')]
    train_features, train_classes, encoder = generate_relational_words_func(
        tables,
        fkg,
        target_table,
        target_attribute,
        relation_order=(1, 1),
        indices=train_index,
        vectorization_type=grid_dict.get('representation_type'),
        num_features=grid_dict.get('num_features'),
        primary_keys=primary_keys)
    test_features, test_classes = generate_relational_words_func(
        tables,
        fkg,
        target_table,
        target_attribute,
        relation_order=(1, 1),
        encoder=encoder,
        indices=test_index,
        vectorization_type=grid_dict.get('representation_type'),
        num_features=grid_dict.get('num_features'),
        primary_keys=primary_keys)

    log_dataset_info(train_features=train_features, test_features=test_features)

    # labels_occurrence_percentage = calculate_positive_class_percentage(
    #     train_classes, test_classes, grid_dict.get('representation_type')
    # )
    # grid_dict['labels_occurrence_percentage'] = labels_occurrence_percentage
    classify_func = CLASSIFIER_FUNC[classifier_name][grid_dict.get('representation_type')]
    try:
        acc, f1, auc_roc, custom_roc_auc = classify_func(args=grid_dict,
                                                         train_features=train_features,
                                                         train_classes=train_classes,
                                                         test_features=test_features,
                                                         test_classes=test_classes)
        return acc, f1, auc_roc, custom_roc_auc
    except Exception as es:
        logging.error(f"Error evaluating classifier {classifier_name}; {es}")
        return None


def main():
    args = parse_arguments()

    setup_directory("tmp")

    config = load_yaml_config('datasets.yaml')

    for dataset in config['datasets']:
        if not dataset.get('enabled', True):
            continue
        process_dataset(dataset=dataset, args=args)


if __name__ == "__main__":
    main()

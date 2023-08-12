import os
import argparse
import csv
import numpy as np

import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from learning import preprocess_and_split
from propositionalization import table_generator, generate_relational_words, generate_custom_relational_words
from classification import *
from utils import save_results

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

feature_func = {"woe": generate_custom_relational_words,
                "sklearn_tfidf": generate_relational_words}

classifier_func = {
    "DRM": {'sklearn_tfidf': prop_drm_tfidf_classification,
            'woe': prop_drm_woe_classification},
    "starspace": {'sklearn_tfidf': prop_star_tfidf_classification,
                  'woe': prop_star_woe_classification},
    "svm_learner": {
        'sklearn_tfidf': traditional_learner_tfidf_classification,
        'woe': traditional_learner_woe_classification
    },
    "extra_tree_learner": {
        'sklearn_tfidf': traditional_learner_tfidf_classification,
        'woe': traditional_learner_woe_classification
    },
    "random_forest_learner": {
        'sklearn_tfidf': traditional_learner_tfidf_classification,
        'woe': traditional_learner_woe_classification
    },
    "ada_boost_learner": {
        'sklearn_tfidf': traditional_learner_tfidf_classification,
        'woe': traditional_learner_woe_classification
    },
    "gradient_boost_learner": {
        'sklearn_tfidf': traditional_learner_tfidf_classification,
        'woe': traditional_learner_woe_classification
    }
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", default="DRM")
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="Learning rate of starspace")
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Number of epochs")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Dropout rate")
    parser.add_argument("--num_features",
                        default=30000,
                        type=int,
                        help="Number of features")
    parser.add_argument("--hidden_size",
                        default=16,
                        type=int,
                        help="Embedding dimension")
    parser.add_argument("--negative_samples_limit",
                        default=10,
                        type=int,
                        help="Max number of negative samples")
    parser.add_argument(
        "--negative_search_limit",
        default=10,
        type=int,
        help=
        "Negative search limit (see starspace docs for extensive description)")
    parser.add_argument(
        "--n_estimators",
        default=16,
        type=int,
        help=
        "No. of estimators used mainly for traditional classifiers")
    parser.add_argument(
        "--kernel",
        default='linear',
        type=str,
        help=
        "Kernel used for SVC")
    parser.add_argument(
        "--C",
        default=1,
        type=int,
        help=
        "Regularization parameter for SVC")
    parser.add_argument(
        "--gamma",
        default='scale',
        help=
        "Kernel coefficient for kernel with value ‘rbf’, used in SVC")
    parser.add_argument(
        "--representation_type",
        default="woe",
        type=str,
        help=
        "Type of representation and weighting. tfidf or binary, also supports scikit's implementations (ordered "
        "patterns)"
    )
    args = parser.parse_args()

    grid_dict = {
        "DRM": {'learner': args.learner, 'epochs': args.epochs, 'learning_rate': args.learning_rate, 'hidden_size': args.hidden_size,
                'dropout': args.dropout, 'representation_type': args.representation_type,
                'num_features': args.num_features},
        "starspace": {'learner': args.learner, 'epochs': args.epochs, 'learning_rate': args.learning_rate,
                      'negative_samples_limit': args.negative_samples_limit, 'hidden_size': args.hidden_size,
                      'negative_search_limit': args.negative_search_limit,
                      'representation_type': args.representation_type,
                      'num_features': args.num_features},
        "svm_learner": {'learner': args.learner, 'epochs': args.epochs, 'representation_type': args.representation_type, 'kernel': args.kernel,
                        'C': args.C, 'gamma': args.gamma},
        "extra_tree_learner": {'learner': args.learner, 'epochs': args.epochs, 'representation_type': args.representation_type,
                               'n_estimators': args.n_estimators},
        "random_forest_learner": {'learner': args.learner, 'epochs': args.epochs, 'representation_type': args.representation_type,
                                  'n_estimators': args.n_estimators},
        "ada_boost_learner": {'learner': args.learner, 'epochs': args.epochs, 'representation_type': args.representation_type,
                              'n_estimators': args.n_estimators},
        "gradient_boost_learner": {'learner': args.learner, 'epochs': args.epochs, 'representation_type': args.representation_type,
                                   'n_estimators': args.n_estimators, 'learning_rate': args.learning_rate}
    }

    variable_types_file = open(
        "variable_types.txt")  # types to be considered.
    variable_types = [
        line.strip().lower() for line in variable_types_file.readlines()
    ]
    variable_types_file.close()
    learner = args.learner
    import os

    # IMPORTANT: a tmp folder must be possible to construct, as the intermediary embeddings are stored here.
    directory = "tmp"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Traverse the data set space
    with open('datasets.txt') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip()[0] != "#":
                line = line.strip().split()
                example_sql = "./sql_data/" + line[0]
                target_table = line[1]
                target_attribute = line[2]

                logging.info("Running for example_sql: " + example_sql +
                             ", target_table: " + target_table +
                             ", target_attribute " + target_attribute)

                tables, fkg, primary_keys = table_generator(
                    example_sql, variable_types)

                # tables[target_table][target_attribute].replace('NULL', np.nan, inplace=True)
                # tables[target_table] = tables[target_table].dropna(axis=0, subset=[target_attribute])
                perf = []
                perf_roc = []
                logging.info("Evaluation of {} - {}".format(
                    grid_dict[args.learner], target_attribute))
                split_gen = preprocess_and_split(
                    tables[target_table],
                    num_fold=10,
                    target_attribute=target_attribute)
                for train_index, test_index in split_gen:
                    # Encoder used only for WoE
                    encoder = CountVectorizer(lowercase=False,
                                              binary=True,
                                              token_pattern='[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-_]+',
                                              ngram_range=(1, 1))  # higher relation orders result in high
                    # memory load, thread with caution!
                    generate_relational_words_func = feature_func[args.representation_type]
                    train_features, train_classes, vectorizer = generate_relational_words_func(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 1),
                        indices=train_index,
                        encoder=encoder,
                        vectorization_type=args.representation_type,
                        num_features=args.num_features,
                        primary_keys=primary_keys)
                    test_features, test_classes = generate_relational_words_func(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 1),
                        vectorizer=vectorizer,
                        indices=test_index,
                        encoder=encoder,
                        vectorization_type=args.representation_type,
                        num_features=args.num_features,
                        primary_keys=primary_keys)

                    classify_func = classifier_func[args.learner][args.representation_type]
                    try:
                        acc, auc_roc = classify_func(args=args,
                                                     train_features=train_features,
                                                     train_classes=train_classes,
                                                     test_features=test_features,
                                                     test_classes=test_classes)
                    except Exception as es:
                        print(es)
                    perf.append(acc)
                    perf_roc.append(auc_roc)
                save_results(args=args, dataset=line[0], accuracies=perf, roc_auc_scores=perf_roc,
                             grid_dict=grid_dict[args.learner])

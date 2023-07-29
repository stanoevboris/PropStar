import numpy as np

import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from learning import preprocess_and_split
from propositionalization import table_generator, generate_relational_words
from classification import prop_drm_classification, prop_star_classification

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

classifier_func = {
    "DRM": prop_drm_classification,
    "starspace": prop_star_classification
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", default="starspace")
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
        "--representation_type",
        default="woe",
        type=str,
        help=
        "Type of representation and weighting. tfidf or binary, also supports scikit's implementations (ordered "
        "patterns)"
    )
    args = parser.parse_args()

    grid_dict = {
        "DRM": [args.epochs, args.learning_rate, args.hidden_size,
                args.dropout, args.representation_type,
                args.num_features],
        "starspace": [args.epochs, args.learning_rate,
                      args.negative_samples_limit, args.hidden_size,
                      args.negative_search_limit, args.representation_type,
                      args.num_features],
        "default": []
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
                    train_features, train_classes, vectorizer = generate_relational_words(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 1),
                        indices=train_index,
                        encoder=encoder,
                        vectorization_type=args.representation_type,
                        num_features=args.num_features)
                    test_features, test_classes = generate_relational_words(
                        tables,
                        fkg,
                        target_table,
                        target_attribute,
                        relation_order=(1, 2),
                        vectorizer=vectorizer,
                        indices=test_index,
                        encoder=encoder,
                        vectorization_type=args.representation_type,
                        num_features=args.num_features)

                    le = preprocessing.LabelEncoder()
                    le.fit(train_classes.values)
                    train_classes = le.transform(train_classes)
                    test_classes = le.transform(test_classes)

                    classify_func = classifier_func[args.learner]
                    acc, auc_roc = classify_func(args=args,
                                                 train_features=train_features,
                                                 train_classes=train_classes,
                                                 test_features=test_features,
                                                 test_classes=test_classes)
                    perf.append(acc)
                    perf_roc.append(auc_roc)
                stx = "|".join(str(x) for x in grid_dict[args.learner])
                mp = np.round(np.mean(perf), 4)
                mp_roc = np.round(np.mean(perf_roc), 4)
                if mp != "nan" and mp != np.nan:
                    print(f"RESULT LINE {args.learner} {mp_roc} {mp} {line[0]} {line[1]} {line[2]}")
                else:
                    pass

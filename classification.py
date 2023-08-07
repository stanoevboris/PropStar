import numpy as np

from neural import *  # DRMs
from learning import *  # starspace

from sklearn.metrics import roc_auc_score
from scipy import sparse
from tqdm import tqdm


def examine_batch_predictions(test_features_indexes,
                              unique_classes,
                              predictions,
                              predictions_scores):
    stats = {}
    for index in tqdm(test_features_indexes):
        stats.setdefault(index, {})
        stats[index] = {label: 0 for label in unique_classes}
        stats[index]["pred_score"] = []
    for index, pred_value, pred_score in tqdm(zip(test_features_indexes, predictions, predictions_scores)):
        stats[index][pred_value] += 1
        stats[index]["pred_score"].append(pred_score)

    batch_preds_classes = []
    batch_preds_scores = []
    for key in stats:
        pred = max(stats[key], key=lambda k: stats[key].get(k) if k in unique_classes else -1)
        pred_score = np.average(stats[key]["pred_score"])
        batch_preds_classes.append(pred)
        batch_preds_scores.append(pred_score)

    return batch_preds_classes, batch_preds_scores


def encode_classes(train_classes, test_classes):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(train_classes)
    train_classes = encoder.transform(train_classes)
    test_classes = encoder.transform(test_classes)
    return train_classes, test_classes, encoder


def prop_drm_tfidf_classification(args, train_features, train_classes, test_features, test_classes):
    train_classes, test_classes, _ = encode_classes(train_classes, test_classes)

    model = E2EDNN(num_epochs=args.epochs,
                   learning_rate=args.learning_rate,
                   hidden_layer_size=args.hidden_size,
                   dropout=args.dropout)

    # standard fit predict
    model.fit(train_features, train_classes)
    predictions = model.predict(test_features)
    acc = accuracy_score(predictions, test_classes)
    logging.info(acc)

    if len(np.unique(test_classes)) == 2:
        predictions = model.predict(test_features,
                                    return_proba=True)
        roc = roc_auc_score(test_classes, predictions)
        logging.info(roc)

    else:
        roc = 0

    return acc, roc


def prop_star_tfidf_classification(args, train_features, train_classes, test_features, test_classes):
    train_classes, test_classes, _ = encode_classes(train_classes, test_classes)

    model = starspaceLearner(epoch=args.epochs,
                             learning_rate=args.learning_rate,
                             neg_search_limit=args.negative_search_limit,
                             dim=args.hidden_size,
                             max_neg_samples=args.negative_samples_limit)

    # standard fit predict
    model.fit(train_features, train_classes)
    predictions = model.predict(test_features,
                                clean_tmp=False)

    if len(predictions) == 0:
        perf_roc = 0
        perf = 0
        return perf, perf_roc

    try:
        acc = accuracy_score(test_classes, predictions)

        logging.info(acc)

        preds_scores = model.predict(
            test_features,
            clean_tmp=True,
            return_int_predictions=False,
            return_scores=True)  # use scores for auc.

        if len(np.unique(test_classes)) == 2:
            roc = roc_auc_score(
                test_classes, preds_scores)
            logging.info(roc)
        else:
            roc = 0
    except Exception as es:
        print(es)
        return

    return acc, roc


def prop_drm_woe_classification(args, train_features, train_classes, test_features, test_classes):
    train_x = sparse.csr_matrix(train_features)
    train_y = [train_classes[index] for index in train_features.index]
    test_x = sparse.csr_matrix(test_features)
    test_y = [test_classes[index] for index in test_features.index]

    train_y, test_y, encoder = encode_classes(train_y, test_y)

    unique_classes = set(test_y)
    logging.info(f"Unique classes:{unique_classes}")

    test_classes_encoded = encoder.transform(list(test_classes.values()))
    model = E2EDNN(num_epochs=args.epochs,
                   learning_rate=args.learning_rate,
                   hidden_layer_size=args.hidden_size,
                   dropout=args.dropout)

    # standard fit predict
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    predictions_scores = model.predict(test_x,
                                       return_proba=True)

    batch_preds_classes, batch_preds_scores = examine_batch_predictions(test_features.index,
                                                                        unique_classes, predictions, predictions_scores)

    acc = accuracy_score(batch_preds_classes, test_classes_encoded)
    logging.info(acc)

    if len(unique_classes) == 2:
        roc = roc_auc_score(test_classes_encoded, batch_preds_scores)
        logging.info(roc)
    else:
        roc = 0

    return acc, roc


def prop_star_woe_classification(args, train_features, train_classes, test_features, test_classes):
    train_x = sparse.csr_matrix(train_features)
    train_y = [train_classes[index] for index in train_features.index]
    test_x = sparse.csr_matrix(test_features)
    test_y = [test_classes[index] for index in test_features.index]

    train_y, test_y, encoder = encode_classes(train_y, test_y)

    unique_classes = set(test_y)
    test_classes_encoded = encoder.transform(list(test_classes.values()))

    model = starspaceLearner(epoch=args.epochs,
                             learning_rate=args.learning_rate,
                             neg_search_limit=args.negative_search_limit,
                             dim=args.hidden_size,
                             max_neg_samples=args.negative_samples_limit)

    # standard fit predict
    model.fit(train_x, train_y)
    try:
        predictions = model.predict(test_x,
                                    clean_tmp=False)

        predictions_scores = model.predict(
            test_x,
            clean_tmp=True,
            return_int_predictions=False,
            return_scores=True)  # use scores for auc.

        batch_preds_classes, batch_preds_scores = examine_batch_predictions(test_features.index,
                                                                            unique_classes,
                                                                            predictions,
                                                                            predictions_scores)
        if len(batch_preds_classes) == 0:
            roc = 0
            acc = 0
            return acc, roc

        acc = accuracy_score(test_classes_encoded, batch_preds_classes)
        logging.info(acc)

        if len(np.unique(test_classes_encoded)) == 2:
            roc = roc_auc_score(
                test_classes_encoded, batch_preds_scores)
            logging.info(roc)
        else:
            roc = 0
    except Exception as es:
        print(es)
        return

    return acc, roc


learners_dict = {"svm_learner": svm_learner,
                 "extra_tree_learner": extra_tree_learner,
                 "random_forest_learner": random_forest_learner,
                 "ada_boost_learner": ada_boost_learner,
                 "gradient_boost_learner": gradient_boost_learner}


def traditional_learner_tfidf_classification(args, train_features, train_classes, test_features, test_classes):
    train_classes, test_classes, _ = encode_classes(train_classes, test_classes)

    learner_func = learners_dict[args.learner]
    model = learner_func(args, train_features, train_classes)

    # standard fit predict
    # model.fit(train_features, train_classes)
    predictions = model.predict(test_features)
    acc = accuracy_score(predictions, test_classes)
    logging.info(acc)

    if len(np.unique(test_classes)) == 2:
        prediction_scores = model.predict_proba(test_features)
        roc = roc_auc_score(test_classes, prediction_scores[:, 1])
        logging.info(roc)

    else:
        roc = 0

    return acc, roc


def traditional_learner_woe_classification(args, train_features, train_classes, test_features, test_classes):
    pass
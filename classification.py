import numpy as np

from neural import *  # DRMs
from learning import *  # starspace

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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
        stats[index]["pred_score"] = {label: [] for label in unique_classes}
    for index, pred_value, pred_score in tqdm(zip(test_features_indexes, predictions, predictions_scores)):
        stats[index][pred_value] += 1
        stats[index]["pred_score"][pred_value].append(pred_score)

    batch_preds_classes = []
    batch_preds_scores = []
    batch_custom_pred_scores = []
    for key in tqdm(stats):
        pred = max(stats[key], key=lambda k: stats[key].get(k) if k in unique_classes else -1)
        pred_score = np.average(stats[key]["pred_score"][pred])

        count_values_by_class = [stats[key][value] for value in unique_classes]
        if 0 in count_values_by_class:
            custom_pred_score = pred_score  # since the other class was not predicted once,
            # we will take the already computed one
        else:
            avg_pred_score_by_class = [np.average(stats[key]["pred_score"][pred_value])
                                       for pred_value in stats[key]["pred_score"]]
            multiplied_list = [a * b for a, b in zip(avg_pred_score_by_class, count_values_by_class)]
            custom_pred_score = np.sum(multiplied_list) / len(unique_classes)

        batch_preds_classes.append(pred)
        batch_preds_scores.append(pred_score)
        batch_custom_pred_scores.append(custom_pred_score)

    return batch_preds_classes, batch_preds_scores, batch_custom_pred_scores


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

    acc = accuracy_score(test_classes, predictions)
    f1 = f1_score(test_classes, predictions)
    logging.info(f'Accuracy:{acc}')
    logging.info(f'F1 score:{f1}')

    if len(np.unique(test_classes)) == 2:
        prediction_scores = model.predict(test_features,
                                          return_proba=True)
        roc = roc_auc_score(test_classes, prediction_scores)
        custom_roc = 0
        logging.info(f'ROC AUC: {roc}')

    else:
        roc = 0
        custom_roc = 0

    return acc, f1, roc, custom_roc


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
        acc = 0
        f1 = 0
        roc = 0
        custom_roc = 0
        return acc, f1, roc, custom_roc

    try:
        acc = accuracy_score(test_classes, predictions)
        f1 = f1_score(test_classes, predictions)
        logging.info(f'Accuracy:{acc}')
        logging.info(f'F1 score:{f1}')

        prediction_scores = model.predict(
            test_features,
            clean_tmp=True,
            return_int_predictions=False,
            return_scores=True)  # use scores for auc.

        if len(np.unique(test_classes)) == 2:
            roc = roc_auc_score(
                test_classes, prediction_scores)
            logging.info(f'ROC AUC: {roc}')
            custom_roc = 0

        else:
            roc = 0
            custom_roc = 0
    except Exception as es:
        print(es)
        return

    return acc, f1, roc, custom_roc


def prop_drm_woe_classification(args, train_features, train_classes, test_features, test_classes):
    try:
        train_y = [train_classes[int(index)] for index in train_features.index]
        test_y = [test_classes[int(index)] for index in test_features.index]
    except Exception as es:
        logging.info("Index is not an integer, continuing without cast")
        print(es)
        train_y = [train_classes[index] for index in train_features.index]
        test_y = [test_classes[index] for index in test_features.index]

    train_x = train_features.copy()
    test_x = test_features.copy()

    train_y, test_y, encoder = encode_classes(train_y, test_y)

    unique_classes = set(test_y)
    logging.info(f"Unique classes:{unique_classes}")
    if len(test_classes) != len(test_features.index):
        test_classes = {key: test_classes[key] for key in test_classes.keys() if key in test_features.index}
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

    batch_preds_classes, batch_pred_scores, batch_custom_pred_scores = examine_batch_predictions(test_features.index,
                                                                                                 unique_classes,
                                                                                                 predictions,
                                                                                                 predictions_scores)

    acc = accuracy_score(test_classes_encoded, batch_preds_classes)
    f1 = f1_score(test_classes_encoded, batch_preds_classes)
    logging.info(f'Accuracy:{acc}')
    logging.info(f'F1 score:{f1}')

    if len(unique_classes) == 2:
        roc = roc_auc_score(test_classes_encoded, batch_pred_scores)
        custom_roc = roc_auc_score(test_classes_encoded, batch_custom_pred_scores)
        logging.info(f'ROC AUC:{roc}')
        logging.info(f'Custom ROC AUC:{custom_roc}')
    else:
        roc = 0
        custom_roc = 0

    return acc, f1, roc, custom_roc


def prop_star_woe_classification(args, train_features, train_classes, test_features, test_classes):
    try:
        train_y = [train_classes[int(index)] for index in train_features.index]
        test_y = [test_classes[int(index)] for index in test_features.index]
    except Exception as es:
        logging.info("Index is not an integer, continuing without cast")
        print(es)
        train_y = [train_classes[index] for index in train_features.index]
        test_y = [test_classes[index] for index in test_features.index]

    train_x = train_features.copy()
    test_x = test_features.copy()

    train_y, test_y, encoder = encode_classes(train_y, test_y)

    unique_classes = set(test_y)
    logging.info(f"Unique classes:{unique_classes}")
    if len(test_classes) != len(test_features.index):
        test_classes = {key: test_classes[key] for key in test_classes.keys() if key in test_features.index}
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

        batch_preds_classes, batch_pred_scores, batch_custom_pred_scores = examine_batch_predictions(
            test_features.index,
            unique_classes,
            predictions,
            predictions_scores)
        if len(batch_preds_classes) == 0:
            acc = 0
            f1 = 0
            roc = 0
            custom_roc = 0
            return acc, f1, roc, custom_roc

        acc = accuracy_score(test_classes_encoded, batch_preds_classes)
        f1 = f1_score(test_classes_encoded, batch_preds_classes)
        logging.info(f'Accuracy:{acc}')
        logging.info(f'F1 score:{f1}')

        if len(unique_classes) == 2:
            roc = roc_auc_score(test_classes_encoded, batch_pred_scores)
            custom_roc = roc_auc_score(test_classes_encoded, batch_custom_pred_scores)
            logging.info(f'ROC AUC:{roc}')
            logging.info(f'Custom ROC AUC:{custom_roc}')
        else:
            roc = 0
            custom_roc = 0
    except Exception as es:
        print(es)
        return

    return acc, f1, roc, custom_roc


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
    predictions = model.predict(test_features)
    acc = accuracy_score(test_classes, predictions)
    f1 = f1_score(test_classes, predictions)
    logging.info(f'Accuracy:{acc}')
    logging.info(f'F1 score:{f1}')

    if len(np.unique(test_classes)) == 2:
        prediction_scores = model.predict_proba(test_features)
        roc = roc_auc_score(test_classes, prediction_scores[:, 1])
        logging.info(f'ROC AUC:{roc}')
        custom_roc = 0  # we are not calculating custom roc for tfidf mode

    else:
        roc = 0
        custom_roc = 0

    return acc, f1, roc, custom_roc


def traditional_learner_woe_classification(args, train_features, train_classes, test_features, test_classes):
    try:
        train_y = [train_classes[int(index)] for index in train_features.index]
        test_y = [test_classes[int(index)] for index in test_features.index]
    except Exception as es:
        logging.info("Index is not an integer, continuing without cast")
        print(es)
        train_y = [train_classes[index] for index in train_features.index]
        test_y = [test_classes[index] for index in test_features.index]

    train_x = train_features.copy()
    test_x = test_features.copy()

    train_y, test_y, encoder = encode_classes(train_y, test_y)

    unique_classes = set(test_y)
    logging.info(f"Unique classes:{unique_classes}")
    if len(test_classes) != len(test_features.index):
        test_classes = {key: test_classes[key] for key in test_classes.keys() if key in test_features.index}
    test_classes_encoded = encoder.transform(list(test_classes.values()))

    learner_func = learners_dict[args.learner]
    model = learner_func(args, train_x, train_y)

    predictions = model.predict(test_x)
    predictions_scores = model.predict_proba(test_x)

    batch_preds_classes, batch_pred_scores, batch_custom_pred_scores = examine_batch_predictions(test_features.index,
                                                                                                 unique_classes,
                                                                                                 predictions,
                                                                                                 predictions_scores[:,
                                                                                                 1])

    acc = accuracy_score(test_classes_encoded, batch_preds_classes)
    f1 = f1_score(test_classes_encoded, batch_preds_classes)
    logging.info(f'Accuracy:{acc}')
    logging.info(f'F1 score:{f1}')

    if len(unique_classes) == 2:
        roc = roc_auc_score(test_classes_encoded, batch_pred_scores)
        custom_roc = roc_auc_score(test_classes_encoded, batch_custom_pred_scores)
        logging.info(f'ROC AUC:{roc}')
        logging.info(f'Custom ROC AUC:{custom_roc}')
    else:
        roc = 0
        custom_roc = 0
    return acc, f1, roc, custom_roc

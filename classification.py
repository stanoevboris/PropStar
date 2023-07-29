from neural import *  # DRMs
from learning import *  # starspace

from sklearn.metrics import roc_auc_score


def prop_drm_classification(args, train_features, train_classes, test_features, test_classes):
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


def prop_star_classification(args, train_features, train_classes, test_features, test_classes):
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

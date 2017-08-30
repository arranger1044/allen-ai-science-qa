from collections import defaultdict

from dataset import load_data_frame
from dataset import get_answers
from dataset import numbers_to_letters
from dataset import get_ids
from dataset import save_predictions
from dataset import load_tensors_preds_from_files
from dataset import load_feature_map
from dataset import load_feature_maps_from_files
from dataset import stack_feature_tensors

from evaluation import compute_accuracy
from evaluation import create_submission
from evaluation import hard_preds
from evaluation import normalize_preds
from evaluation import min_max_normalize_preds
from evaluation import softmax_normalize_preds

from ensemble import collect_predictions_from_dirs
from ensemble import predictions_paths_from_dirs
from ensemble import augment_predictions
from ensemble import aggregate_function_factory
from ensemble import OneVsRestClassifier
from ensemble import OneVsOneClassifier
from ensemble import OneVsOneClassifierDiff
from ensemble import OneVsRestClassifierDiff

from ensemble import feature_importances_meta_clf
from ensemble import feature_importances_meta_clf_cv

from embeddings import data_frame_to_matrix, data_frame_to_matrix_q_plus_a
from embeddings import load_word2vec_model
from embeddings import sum_sentence_embedding

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold


from sklearn import decomposition
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import neighbors

from sklearn import cross_validation

from ensemble import Calibrator

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2

import pickle
import json

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


import numpy

import random

import datetime

import itertools

import os

import logging

from xgboost_l2r_exp import train_simple_xgboost
from xgboost_l2r_exp import eval_xgboost

from ranklib import pair_matrix_to_ranklib_format
from ranklib import ranklib_train_eval
from ranklib import ranklib_eval

TRAIN_SUBS_EXT = 'train.submission'
VALID_SUBS_EXT = 'valid.submission'

TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'

META_CLASS_DICT = {'ovr': OneVsRestClassifier,
                   'ovo': OneVsOneClassifier,
                   'ovod': OneVsOneClassifierDiff,
                   'ovrd': OneVsRestClassifierDiff}

LEARNER_DICT = {
    'rfc': ensemble.RandomForestClassifier,
    'abc': ensemble.AdaBoostClassifier,
    'etc': ensemble.ExtraTreesClassifier,
    'gbm': ensemble.GradientBoostingClassifier,
    'lr': linear_model.LogisticRegression,
    'svc': svm.SVC,
    'nusvc': svm.NuSVC,
    'gnb': naive_bayes.GaussianNB,
    'lda': LinearDiscriminantAnalysis,
    'qda': QuadraticDiscriminantAnalysis,
    'dt': tree.DecisionTreeClassifier,
    'knn': neighbors.KNeighborsClassifier}

RAND_LEARNERS = {'rfc', 'abc', 'etc', 'gbm', 'lr',
                 'svc', 'nusvc', 'dt'}
RAND_LEARNERS_CLS = {learner_class for name, learner_class in LEARNER_DICT.items()
                     if name in RAND_LEARNERS}


def compute_aggregation(preds, aggr_func_type):
    # print('Dealing with pred matrix of {}'.format(preds.shape))

    aggr_func = aggregate_function_factory(aggr_func_type)
    # aggr_func = AGGR_FUNC_DICT[aggr_func_type]

    aggr_preds = aggr_func(preds)

    assert len(aggr_preds) == len(preds)
    # print('Reduced to', aggr_preds.shape)

    return aggr_preds


def make_submission(aggr_preds, aggr_func_type, output, id, dataset, ids=None):

    if dataset == 'train':
        ext = TRAIN_SUBS_EXT
    elif dataset == 'valid':
        ext = VALID_SUBS_EXT

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aggr_preds_file_name = str(id) + '_' + aggr_func_type + '_' + \
        date_str + '_' + ext
    output_path = os.path.join(output, id)
    aggr_preds_out_path = os.path.join(output_path, aggr_preds_file_name)

    os.makedirs(output_path, exist_ok=True)

    letter_preds = numbers_to_letters(aggr_preds)
    create_submission(letter_preds, aggr_preds_out_path, ids=ids)


def make_scores(aggr_preds, aggr_func_type, output, id, dataset, ids=None):

    if dataset == 'train':
        ext = TRAIN_PREDS_EXT
    elif dataset == 'valid':
        ext = VALID_PREDS_EXT

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aggr_preds_file_name = str(id) + '_' + aggr_func_type + '_' + \
        date_str + '_' + ext
    output_path = os.path.join(output, id)
    aggr_preds_out_path = os.path.join(output_path, aggr_preds_file_name)

    os.makedirs(output_path, exist_ok=True)

    save_predictions(aggr_preds, aggr_preds_out_path, ids=ids)


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument('preds', type=str, nargs='+',
                        help='One or more dirs to retrieve the paths')

    parser.add_argument('-t', '--train', type=str,
                        default='../data/training_set.tsv',
                        help='Training set path')

    # parser.add_argument('-x', '--split-file', type=str, nargs='?',
    #                     default=None,
    #                     help='Split file procedure')

    parser.add_argument('-z', '--n-folds', type=int, nargs='?',
                        default=5,
                        help='Number of folds for cv')

    parser.add_argument('-f', '--test', type=str,
                        default='../data/validation_set.tsv',
                        help='Test set path')

    parser.add_argument('-c', '--meta-classifier', type=str,
                        default='ovr',
                        help='Meta classifier type (ovr|ovo)')

    parser.add_argument('-m', '--base-models', type=str,
                        default='base_models.json',
                        help='Path to JSON file containing base models configs')

    parser.add_argument('-s', '--stacking-base-model', type=str,
                        default='stacking_base_model.json',
                        help='Path to JSON file containing stacking base model config')

    parser.add_argument('--cv', action='store_true',
                        help='Cross validating model')

    parser.add_argument('--seed', type=int, nargs='+',
                        default=[1337],
                        help='Seed for the random generator')

    parser.add_argument('-j', '--transforms', type=str, nargs='+',
                        default=['log1p'],
                        help='Transforms to apply to target var')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=0,
                        help='Verbosity level')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/onevs/',
                        help='Output dir path')

    parser.add_argument('--aug', type=str, nargs='?',
                        default=None,
                        help='Augment features (log|ens|len)')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    parser.add_argument('-a', '--aggr-func',  type=str, nargs='+',
                        default=['avg'],
                        help='Aggregating functions (avg, maj)')

    parser.add_argument('--train-ext', type=str,
                        default=TRAIN_PREDS_EXT,
                        help='Train preds file extension')

    parser.add_argument('--valid-ext', type=str,
                        default=VALID_PREDS_EXT,
                        help='Valid preds file extension')

    parser.add_argument('--id',  type=str,
                        default='0x00',
                        help='Counter to identify the output')

    parser.add_argument('--load-matrices', action='store_true',
                        help='Load numpy matrices instead of predictions')

    parser.add_argument('--aug-features', type=str, nargs='+',
                        help='Augment features by passing feature map files paths')

    parser.add_argument('--calibrate', action='store_true',
                        help='Whether to use a logistic calibrator before learning a meta classifier')

    parser.add_argument('--use-seeds-stacking', action='store_true',
                        help='''Whether to use different seeds results as stacking predictors
                        (otherwise they get averaged)''')

    parser.add_argument('--feature-selection', type=float, nargs='?',
                        default=None,
                        help='Feature selection threshold to apply to feature importances')

    #
    # parsing the args
    args = parser.parse_args()
    print(args)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)
    # I shall print here all the stats

    n_answers = 4

    #
    # loading frames, labels, ids
    train_frame = load_data_frame(args.train)
    valid_frame = load_data_frame(args.test)

    train_labels = numpy.array(get_answers(train_frame, numeric=True))
    # train_labels = vec_to_pair_vec(train_labels_vec, n_answers)

    train_frame_ids = numpy.array(get_ids(train_frame))
    valid_frame_ids = numpy.array(get_ids(valid_frame))

    #
    # loading preds tensors
    if args.load_matrices:
        print('By appending numpy files')
        train_preds = load_tensors_preds_from_files(args.preds, args.train_ext)
        valid_preds = load_tensors_preds_from_files(args.preds, args.valid_ext)
    else:
        print('By scanning directories')
        train_preds = collect_predictions_from_dirs(args.preds, args.train_ext)
        valid_preds = collect_predictions_from_dirs(args.preds, args.valid_ext)

    print('Retrieved TRAIN', train_preds.shape)
    print('Retrieved VALID', valid_preds.shape)

    if args.aug == 'log':
        train_preds = augment_predictions(train_preds, aggr_funcs=[], normalize=False, logify=True)
        valid_preds = augment_predictions(valid_preds, aggr_funcs=[], normalize=False, logify=True)
    elif args.aug == 'ens':
        train_preds = augment_predictions(train_preds, normalize=False)
        valid_preds = augment_predictions(valid_preds, normalize=False)
    elif args.aug == 'len':
        train_preds = augment_predictions(train_preds,  normalize=False, logify=True)
        valid_preds = augment_predictions(valid_preds,  normalize=False, logify=True)
    elif args.aug == 'loo':
        train_preds = numpy.log1p(train_preds)
        valid_preds = numpy.log1p(valid_preds)
    elif args.aug is not None:
        try:
            aug_funcs = args.aug.split(':')
            train_preds = augment_predictions(train_preds, aggr_funcs=aug_funcs, normalize=False)
            valid_preds = augment_predictions(valid_preds, aggr_funcs=aug_funcs, normalize=False)
        except:
            raise ValueError('Unrecognized aug func options')

    print('Processed TRAIN', train_preds.shape)
    print('Processed VALID', valid_preds.shape)

    meta_classifier = META_CLASS_DICT[args.meta_classifier]
    print('Using meta classifier: {}'.format(args.meta_classifier))

    # augment_features = True
    if args.aug_features:

        print('\n\nAAUGMENTING FEATURES')
        # train_feature_map = load_feature_map('../features/train.rule')
        # valid_feature_map = load_feature_map('../features/valid.rule')
        train_feature_maps = load_feature_maps_from_files(args.aug_features, args.train_ext)
        valid_feature_maps = load_feature_maps_from_files(args.aug_features, args.valid_ext)

        print('Loaded more features train:', sum([map.shape[2] for map in train_feature_maps]))
        print('Loaded more features valid:', sum([map.shape[2] for map in valid_feature_maps]))

        # train_preds = numpy.dstack((train_preds, train_feature_map))
        # valid_preds = numpy.dstack((valid_preds, valid_feature_map))
        train_preds = stack_feature_tensors(train_preds, train_feature_maps)
        valid_preds = stack_feature_tensors(valid_preds, valid_feature_maps)
        print('Tot features', train_preds.shape)
        print('Tot features', valid_preds.shape)

    # feature_selection = True

    if args.feature_selection:
        threshold = args.feature_selection
        # model = meta_classifier(LEARNER_DICT['lr'],
        #                         feature_selector={},  # base_feature_sel_dict,
        #                         **{
        #                             "fit_intercept": True,
        #                             "class_weight": "balanced",
        #                             "penalty": "l2",
        #                             "C": 10.0,
        #                             "max_iter": 200})
        # model.fit(train_preds, train_labels)
        # if hasattr(model, 'feature_importances_'):
        #     print('using feature_importances_')
        #     feature_importances = model.feature_importances_

        # feature_importances = numpy.abs(feature_importances)

        feature_importances = \
            feature_importances_meta_clf(train_preds,
                                         train_labels,
                                         meta_classifier=OneVsRestClassifier,
                                         model_dict={
                                             "base_model": linear_model.LogisticRegression,
                                             "base_model_params": {
                                                 "fit_intercept": True,
                                                 "class_weight": "balanced",
                                                 "penalty": "l2",
                                                 "C": 10.0,
                                                 "max_iter": 200}
                                         }
                                         # model_dict={
                                         #     "base_model": ensemble.RandomForestClassifier,
                                         #     "base_model_params": {
                                         #         "n_estimators": 50,
                                         #         "class_weight": "balanced",
                                         #         "max_depth": 4,
                                         #         "n_jobs": -1,
                                         #         "criterion": "gini"}
                                         # }
                                         )
        feature_importances = ((feature_importances - numpy.min(feature_importances)) /
                               (numpy.max(feature_importances) - numpy.min(feature_importances)))

        # feature_importances = \
        #     feature_importances_meta_clf_cv(train_preds,
        #                                     train_labels,
        #                                     seeds=[1337],
        #                                     n_folds=5,
        #                                     meta_classifier=OneVsRestClassifier,
        #                                     model_dict={
        #                                         "base_model": linear_model.LogisticRegression,
        #                                         "base_model_params": {
        #                                             "fit_intercept": True,
        #                                             "class_weight": "balanced",
        #                                             "penalty": "l2",
        #                                             "C": 10.0,
        #                                             "max_iter": 200}
        #                                     })

        print('FEATURE IMPORTANCES', feature_importances)
        print('SORTED', numpy.sort(feature_importances))
        sorted_feature_importances = numpy.argsort(feature_importances)[::-1]
        print('SORTED ARGS', sorted_feature_importances)
        selected_features = feature_importances > threshold
        print(sum(selected_features))

        if not args.load_matrices:
            preds_paths = predictions_paths_from_dirs(args.preds, args.train_ext)
            ordered_preds_paths = numpy.array(preds_paths)[sorted_feature_importances]
            selected_preds_paths = numpy.array(preds_paths)[selected_features]
            print('\n\nORDERED PREDS')
            print('\n'.join('{0}, {1}'.format(score, path)
                            for score, path in zip(numpy.sort(feature_importances)[::-1],
                                                   ordered_preds_paths)))
            print('\n\nSELECTED PREDS')
            print('\n'.join(path for path in sorted(selected_preds_paths)))
        train_preds = train_preds[:, :, selected_features]
        valid_preds = valid_preds[:, :, selected_features]
        print('Processed TRAIN', train_preds.shape)
        print('Processed VALID', valid_preds.shape)

    #
    # Opening the file for test prediction
    #
    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = None
    if args.exp_name:
        out_path = os.path.join(args.output, 'exp_' + args.exp_name)
    else:
        out_path = os.path.join(args.output, 'exp_' + date_string)
    out_log_path = os.path.join(out_path, '/exp.log')
    os.makedirs(out_path, exist_ok=True)

    print(out_path)

    silent = 1 if args.verbose < 1 else 0

    # #
    # # creating dir if non-existant
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    base_models_str = """
    [
        ["logistic1", {
            "base_model": "lr",
            "base_model_params": {
                "fit_intercept": true,
                "class_weight": "balanced",
                "penalty": "l2",
                "C": 10.0,
                "max_iter": 200}
        }],
        ["randfor1", {
            "base_model": "rfc",
            "base_model_params": {
                "n_estimators": 30,
                "class_weight": "balanced",
                "max_depth": 4,
                "n_jobs": -1,
                "criterion": "gini"}
        }],
        ["adaboost1", {
            "base_model": "abc",
            "base_model_params": {"n_estimators": 60,
                                  "learning_rate": 0.1}
        }],
        ["extrtree1", {
            "base_model": "etc",
            "base_model_params": {
                "n_estimators": 120,
                "class_weight": "balanced",
                "max_depth": 4,
                "n_jobs": -1,
                "criterion": "gini"}
        }],
        ["gradboost1", {
            "base_model": "gbm",
            "base_model_params": {"loss": "deviance",
                                  "learning_rate": 0.1,
                                  "n_estimators": 30,
                                  "subsample": 1.0,
                                  "min_samples_split": 2,
                                  "min_samples_leaf": 1,
                                  "min_weight_fraction_leaf": 0.0,
                                  "max_depth": 3}
        }]
    ]
    """

    if args.base_models is not None:
        if args.base_models.lower() == 'none':
            args.base_models = None

    models = None
    if args.base_models is not None:
        with open(args.base_models, 'r') as json_models:
            models = json.load(json_models)
    else:
        models = json.loads(base_models_str)

    #
    # mapping model strings to sklearn classes
    for model_name, model_dict in models:
        model_dict['base_model'] = LEARNER_DICT[model_dict['base_model']]

    #
    # generating folds
    n_folds = args.n_folds

    cv_out_path = os.path.join(out_path, 'cv')
    os.makedirs(cv_out_path, exist_ok=True)

    calibration = False

    train_acc_list = defaultdict(list)
    valid_acc_list = defaultdict(list)

    out_log_path = os.path.join(out_path, 'exp.log')

    n_models = len(models)
    n_seeds = len(args.seed)

    #
    # stacking
    #
    train_stacked_preds = numpy.zeros((train_preds.shape[0],
                                       n_answers,
                                       n_models,
                                       n_seeds))
    valid_stacked_preds = numpy.zeros((valid_preds.shape[0],
                                       n_answers,
                                       n_models,
                                       n_folds,
                                       n_seeds))

    #
    # calibrator
    normalizers = [min_max_normalize_preds, softmax_normalize_preds]
    caliber_base_model = linear_model.LogisticRegression
    caliber_base_model_params = {'fit_intercept': True,
                                 'class_weight': 'balanced',
                                 'penalty': 'l2',
                                 'C': 10.0}

    #
    # for storing folds accs
    fold_train_preds_list = [[None for i in range(n_folds)] for j in range(n_seeds)]
    fold_test_preds_list = [[None for i in range(n_folds)] for j in range(n_seeds)]

    # pca_transformer = {'base_transformer': decomposition.PCA,
    #                    'base_transformer_params': {'n_components': 'mle',
    #                                                'whiten': False}}

    print('STACKING TRAIN shape', train_stacked_preds.shape)
    print('STACKING VALID shape', valid_stacked_preds.shape)

    with open(out_log_path, 'w') as out_log:

        out_log.write('{}\n'.format(args))
        out_log.flush()

        #
        # looping all over seeds
        for r, seed in enumerate(args.seed):

            print('\n**  Processing seed: {0} **'.format(seed))
            out_log.write('\n\n**  Processing seed: {0} **\n'.format(seed))
            out_log.flush()

            # kf = KFold(train_preds.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
            kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

            rand_gen = random.Random(seed)
            numpy_rand_gen = numpy.random.RandomState(seed)
            numpy.random.seed(seed)

            for m, (model_name, model_dict) in enumerate(models):

                if model_dict['base_model'] in RAND_LEARNERS_CLS:
                    print('setting seed')
                    model_dict['base_model_params']['random_state'] = seed

                print('\n>> model: {0} <<'.format(model_name))
                out_log.write('\n>> model: {0} <<\n'.format(model_name))
                out_log.flush()

                model_out_path = os.path.join(cv_out_path, '{}'.format(model_name))
                os.makedirs(model_out_path, exist_ok=True)

                seed_out_path = os.path.join(model_out_path, '{}'.format(seed))
                os.makedirs(seed_out_path, exist_ok=True)

                cv_valid_accs = []
                cv_train_accs = []

                for k, (train_ids, test_ids) in enumerate(kf):

                    fold_out_path = os.path.join(seed_out_path, '{}'.format(k))
                    os.makedirs(fold_out_path, exist_ok=True)

                    print('Fold', k)

                    train_x = train_preds[train_ids]
                    train_y = train_labels[train_ids]
                    train_ids = train_frame_ids[train_ids]
                    test_x = train_preds[test_ids]
                    test_y = train_labels[test_ids]
                    test_ids_f = train_frame_ids[test_ids]

                    #
                    # calibrate preds?
                    if args.calibrate:
                        caliber = Calibrator(caliber_base_model,
                                             normalizers=normalizers,
                                             **caliber_base_model_params)
                        caliber.fit(train_x, train_y)
                        train_x = caliber.predict(train_x)

                    model = meta_classifier(model_dict['base_model'],
                                            feature_selector=None,  # base_feature_sel_dict,
                                            **model_dict['base_model_params'])

                    #
                    # fitting
                    # print('Fitting')
                    model.fit(train_x, train_y)

                    #
                    # predicting
                    # print('Predicting on train')
                    train_pred_probs = model.predict(train_x)
                    hard_train_preds = hard_preds(train_pred_probs)
                    train_preds_path = os.path.join(fold_out_path, '{0}.{1}'.format(k,
                                                                                    TRAIN_PREDS_EXT))
                    # normalizing
                    # train_pred_probs = train_pred_probs / numpy.sum(train_pred_probs,
                    #                                                 axis=1,
                    #                                                 keepdims=True)
                    save_predictions(train_pred_probs, train_preds_path, train_ids)

                    train_acc = compute_accuracy(train_y, hard_train_preds)
                    print('ON TRAIN', train_acc)
                    cv_train_accs.append(train_acc)

                    #
                    # predicting
                    # print('Predicting on test')
                    test_pred_probs = model.predict(test_x)
                    hard_test_preds = hard_preds(test_pred_probs)
                    test_preds_path = os.path.join(fold_out_path, '{0}.{1}'.format(k,
                                                                                   VALID_PREDS_EXT))
                    #
                    # normalizing
                    # test_pred_probs = test_pred_probs / numpy.sum(test_pred_probs,
                    #                                               axis=1,
                    #                                               keepdims=True)
                    save_predictions(test_pred_probs, test_preds_path, test_ids_f)

                    test_acc = compute_accuracy(test_y, hard_test_preds)
                    print('ON TEST', test_acc)
                    cv_valid_accs.append(test_acc)

                    #
                    # predicting on the whole valid
                    whole_valid_preds = model.predict(valid_preds)

                    #
                    # storing for aggregation
                    if fold_train_preds_list[r][k] is None:
                        fold_train_preds_list[r][k] = train_pred_probs
                    else:
                        fold_train_preds_list[r][k] = numpy.dstack((fold_train_preds_list[r][k],
                                                                    train_pred_probs))

                    # print('T-SHAPE', train_pred_probs.shape, r,
                    #       k, fold_train_preds_list[r][k].shape)

                    if fold_test_preds_list[r][k] is None:
                        fold_test_preds_list[r][k] = test_pred_probs
                    else:
                        fold_test_preds_list[r][k] = numpy.dstack((fold_test_preds_list[r][k],
                                                                   test_pred_probs))

                    #
                    # storing for stacking
                    train_stacked_preds[test_ids, :, m, r] = test_pred_probs
                    valid_stacked_preds[:, :, m, k, r] = whole_valid_preds

                avg_train_acc = sum(cv_train_accs) / n_folds
                print('AVG on TRAIN', avg_train_acc)
                out_log.write('AVG on TRAIN {}\n'.format(avg_train_acc))
                avg_valid_acc = sum(cv_valid_accs) / n_folds
                print('AVG on TEST', avg_valid_acc)
                out_log.write('AVG on TEST {}\n'.format(avg_valid_acc))
                out_log.flush()

                train_acc_list[model_name].append(avg_train_acc)
                valid_acc_list[model_name].append(avg_valid_acc)

            print('\n')

        #
        # averaging over all seeds
        print('\nTRAIN RES:')
        out_log.write('\nTRAIN RES:\n')
        for model_name, model_accs in sorted(train_acc_list.items(), key=lambda t: t[1]):
            model_avg_acc = sum(model_accs) / len(args.seed)
            print('\t{0}:\t{1}'.format(model_name, model_avg_acc))
            out_log.write('\t{0}:\t{1}\n'.format(model_name, model_avg_acc))
            out_log.flush()

        print('\nTEST RES:')
        out_log.write('\nTEST RES:\n')
        for model_name, model_accs in sorted(valid_acc_list.items(), key=lambda t: t[1]):
            model_avg_acc = sum(model_accs) / len(args.seed)
            print('\t{0}:\t{1}'.format(model_name, model_avg_acc))
            out_log.write('\t{0}:\t{1}\n'.format(model_name, model_avg_acc))
            out_log.flush()

        train_preds_list = None
        test_preds_list = None

        #
        # SINGLE MODELS
        print('\nSINGLE EVALS')
        for model_name, model_dict in models:

            model_out_path = os.path.join(out_path, '{}'.format(model_name))
            os.makedirs(model_out_path, exist_ok=True)

            print(model_name)

            model = meta_classifier(model_dict['base_model'],
                                    feature_selector=None,  # base_feature_sel_dict,
                                    **model_dict['base_model_params'])
            #
            # fit on whole preds
            model.fit(train_preds, train_labels)

            #
            # saving
            model_file_path = os.path.join(model_out_path, model_name + '.model')
            with open(model_file_path, 'wb') as model_file:
                pickle.dump(model, model_file)

            #
            # predicting
            whole_train_preds = model.predict(train_preds)
            hard_whole_train_preds = hard_preds(whole_train_preds)
            train_preds_path = os.path.join(model_out_path, '{0}_{1}'.format(model_name,
                                                                             TRAIN_PREDS_EXT))
            # whole_train_preds = whole_train_preds / numpy.sum(whole_train_preds,
            #                                                   axis=1,
            #                                                   keepdims=True)
            save_predictions(whole_train_preds, train_preds_path, train_frame_ids)
            train_preds_path = os.path.join(model_out_path, '{0}_{1}'.format(model_name,
                                                                             TRAIN_SUBS_EXT))
            create_submission(numbers_to_letters(hard_whole_train_preds),
                              output=train_preds_path,
                              ids=train_frame_ids)

            whole_valid_preds = model.predict(valid_preds)
            hard_whole_valid_preds = hard_preds(whole_valid_preds)
            valid_preds_path = os.path.join(model_out_path, '{0}_{1}'.format(model_name,
                                                                             VALID_PREDS_EXT))
            # whole_valid_preds = whole_valid_preds / numpy.sum(whole_valid_preds,
            #                                                   axis=1,
            #                                                   keepdims=True)
            save_predictions(whole_valid_preds, valid_preds_path, valid_frame_ids)
            valid_preds_path = os.path.join(model_out_path, '{0}_{1}'.format(model_name,
                                                                             VALID_SUBS_EXT))
            create_submission(numbers_to_letters(hard_whole_valid_preds),
                              output=valid_preds_path,
                              ids=valid_frame_ids)

            if train_preds_list is None:
                train_preds_list = whole_train_preds
            else:
                train_preds_list = numpy.dstack((train_preds_list,
                                                 whole_train_preds))

            if test_preds_list is None:
                test_preds_list = whole_valid_preds
            else:
                test_preds_list = numpy.dstack((test_preds_list,
                                                whole_valid_preds))
        #
        # AGGREGATING
        #
        if args.aggr_func:
            print('\n\nAGGREGATING\n')
            out_log.write('\n\nAGGREGATING\n')

            aggr_out_path = os.path.join(cv_out_path, 'aggr')
            os.makedirs(aggr_out_path, exist_ok=True)

            for aggr_func in args.aggr_func:

                print('\nAggregating with', aggr_func)
                out_log.write(aggr_func + '\n')

                avg_aggrs_train = []
                avg_aggrs_test = []

                # print('LENS', len(fold_test_preds_list), len(
                #     fold_test_preds_list[0]), fold_test_preds_list[0][0].shape)
                for r, seed in enumerate(args.seed):

                    kf = StratifiedKFold(train_labels,
                                         n_folds=n_folds, shuffle=True, random_state=seed)

                    train_fold_accs = []
                    test_fold_accs = []

                    seed_out_path = os.path.join(aggr_out_path, '{}'.format(seed))
                    os.makedirs(seed_out_path, exist_ok=True)

                    print('\tSeed: {}'.format(seed))
                    out_log.write('\tSeed: {}\n'.format(seed))

                    for k, (t_ids, v_ids) in enumerate(kf):

                        fold_test_preds = fold_test_preds_list[r][k]
                        print(fold_test_preds.shape, r, k)
                        test_aggr_preds = compute_aggregation(fold_test_preds, aggr_func)

                        fold_train_preds = fold_train_preds_list[r][k]
                        train_aggr_preds = compute_aggregation(fold_train_preds, aggr_func)

                        if aggr_func == 'maj':
                            hard_test_aggr_preds = test_aggr_preds
                            hard_train_aggr_preds = train_aggr_preds
                        else:
                            hard_train_aggr_preds = hard_preds(train_aggr_preds)
                            norm_train_aggr_preds = normalize_preds(train_aggr_preds,
                                                                    cos_sim=False,
                                                                    eps=1e-10)
                            make_scores(norm_train_aggr_preds,
                                        aggr_func,
                                        seed_out_path,
                                        args.id,
                                        'train',
                                        train_frame_ids[t_ids])
                            make_submission(hard_train_aggr_preds,
                                            aggr_func,
                                            seed_out_path,
                                            args.id,
                                            'train',
                                            train_frame_ids[t_ids])

                            avg_train_acc = compute_accuracy(train_labels[t_ids],
                                                             hard_train_aggr_preds)
                            train_fold_accs.append(avg_train_acc)
                            print('\tTRAIN:', avg_train_acc)

                            hard_test_aggr_preds = hard_preds(test_aggr_preds)
                            norm_test_aggr_preds = normalize_preds(test_aggr_preds,
                                                                   cos_sim=False,
                                                                   eps=1e-10)
                            make_scores(norm_test_aggr_preds,
                                        aggr_func,
                                        seed_out_path,
                                        args.id,
                                        'valid',
                                        train_frame_ids[v_ids])
                            make_submission(hard_test_aggr_preds,
                                            aggr_func,
                                            seed_out_path,
                                            args.id,
                                            'valid',
                                            train_frame_ids[v_ids])

                            avg_test_acc = compute_accuracy(train_labels[v_ids],
                                                            hard_test_aggr_preds)
                            test_fold_accs.append(avg_test_acc)
                            print('\tTEST:', avg_test_acc)

                    print('\t\tTRAIN folds', '\t'.join(str(f) for f in train_fold_accs))
                    out_log.write('\t\tTRAIN folds' + '\t'.join(str(f)
                                                                for f in train_fold_accs) + '\n')
                    print('\t\tTEST folds', '\t'.join(str(f) for f in test_fold_accs))
                    out_log.write('\t\tTEST folds' + '\t'.join(str(f)
                                                               for f in test_fold_accs) + '\n')
                    avg_train = sum(train_fold_accs) / n_folds
                    avg_test = sum(test_fold_accs) / n_folds
                    print('\t\t\tAGG TRAIN', avg_train)
                    print('\t\t\tAGG TEST', avg_test)

                    avg_aggrs_train.append(avg_train)
                    avg_aggrs_test.append(avg_test)

                    out_log.write('\tAGG TRAIN {}\n'.format(avg_train))
                    out_log.write('\tAGG TEST {}\n'.format(avg_test))
                    out_log.flush()

                print('\nAVG over seeds:')
                print('AVG TRAIN', '\t'.join(str(f) for f in avg_aggrs_train))
                out_log.write('\tAVG TRAIN' + '\t'.join(str(f)
                                                        for f in avg_aggrs_train) + '\n')
                print('AVG TEST', '\t'.join(str(f) for f in avg_aggrs_test))
                out_log.write('\tAVG TEST' + '\t'.join(str(f)
                                                       for f in avg_aggrs_test) + '\n')
                avg_train_all = sum(avg_aggrs_train) / n_seeds
                avg_test_all = sum(avg_aggrs_test) / n_seeds
                print('AGG TRAIN ALL', avg_train_all)
                print('AGG TEST ALL', avg_test_all)

                out_log.write('\tAGG TRAIN ALL {}\n'.format(avg_train_all))
                out_log.write('\tAGG TEST ALL {}\n'.format(avg_test_all))
                out_log.flush()

        #
        # STACKING
        #
        print('\n\nSTACKING\n')
        out_log.write('\n\nSTACKING\n')

        if args.use_seeds_stacking:
            train_stacked_preds = train_stacked_preds.reshape(train_stacked_preds.shape[0],
                                                              train_stacked_preds.shape[1],
                                                              -1)
            valid_stacked_preds = numpy.mean(valid_stacked_preds, axis=3)
            valid_stacked_preds = valid_stacked_preds.reshape(valid_stacked_preds.shape[0],
                                                              valid_stacked_preds.shape[1],
                                                              -1)
        else:
            #
            # first aggregate on seeds
            train_stacked_preds = numpy.mean(train_stacked_preds, axis=3)
            valid_stacked_preds = numpy.mean(valid_stacked_preds, axis=4)

            #
            # then aggregate stacked preds for valid on folds
            valid_stacked_preds = numpy.mean(valid_stacked_preds, axis=3)
        print('VALID stacked shape', valid_stacked_preds.shape)

        stack_out_path = os.path.join(out_path, 'stacking')
        os.makedirs(stack_out_path, exist_ok=True)
        stack_out_train_path = os.path.join(stack_out_path, 'train.stacking')
        stack_out_valid_path = os.path.join(stack_out_path, 'valid.stacking')
        numpy.save(stack_out_train_path, train_stacked_preds)
        numpy.save(stack_out_valid_path, valid_stacked_preds)

        stacking_base_model = linear_model.LogisticRegression
        stacking_base_model_params = {'fit_intercept': True,
                                      'class_weight': 'balanced',
                                      'penalty': 'l2',
                                      # 'C': 10.0,
                                      'C': 10.0,
                                      'random_state': 1337,
                                      'max_iter': 200}

        train_acc_list = []
        test_acc_list = []

        for r, seed in enumerate(args.seed):

            print('\nSeed:{}'.format(seed))
            out_log.write('\nSeed:{}\n'.format(seed))

            kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)
            numpy.random.seed(seed)
            stacking_base_model_params['random_state'] = seed

            train_accs = []
            test_accs = []
            for k, (train_ids, test_ids) in enumerate(kf):
                print('Stacking Fold', k)

                train_x = train_stacked_preds[train_ids]
                train_y = train_labels[train_ids]
                test_x = train_stacked_preds[test_ids]
                test_y = train_labels[test_ids]

                # if calibration:
                #     caliber = Calibrator(caliber_base_model, **caliber_base_model_params)
                #     caliber.fit(train_x, train_y)
                #     train_x = caliber.predict(train_x)

                model = meta_classifier(stacking_base_model, **stacking_base_model_params)

                #
                # fitting
                # print('Fitting')
                model.fit(train_x, train_y)

                #
                # predicting
                # print('Predicting on train')
                train_pred_probs = model.predict(train_x)

                hard_train_preds = hard_preds(train_pred_probs)

                train_acc = compute_accuracy(train_y, hard_train_preds)
                print('ON TRAIN', train_acc)

                #
                # predicting
                # print('Predicting on test')
                test_pred_probs = model.predict(test_x)

                hard_test_preds = hard_preds(test_pred_probs)

                test_acc = compute_accuracy(test_y, hard_test_preds)
                print('ON TEST', test_acc)

                train_accs.append(train_acc)
                test_accs.append(test_acc)

            print('TRAIN ACCS', train_accs)
            print('TEST ACCS', test_accs)
            out_log.write('TRAIN ACCS {}\n'.format(train_accs))
            out_log.write('TEST ACCS {}\n'.format(test_accs))

            avg_train_acc = sum(train_accs) / n_folds
            train_acc_list.append(avg_train_acc)

            avg_test_acc = sum(test_accs) / n_folds
            test_acc_list.append(avg_test_acc)

            print('AVG TRAIN', avg_train_acc)
            print('AVG VALID', avg_test_acc)
            out_log.write('AVG TRAIN {}\n'.format(avg_train_acc))
            out_log.write('AVG VALID {}\n'.format(avg_test_acc))
            out_log.flush()

        final_avg_train_acc = sum(train_acc_list) / n_seeds
        final_avg_test_acc = sum(test_acc_list) / n_seeds
        print('\nfinal STACKING AVG TRAIN {}'.format(final_avg_train_acc))
        print('\nfinal STACKING AVG TEST {}'.format(final_avg_test_acc))
        out_log.write('\nfinal STACKING AVG TRAIN {}\n'.format(final_avg_train_acc))
        out_log.write('\nfinal STACKING AVG TEST {}\n'.format(final_avg_test_acc))

        stacked_model = meta_classifier(stacking_base_model, **stacking_base_model_params)

        stacked_model.fit(train_stacked_preds, train_labels)

        stacked_model_file = os.path.join(stack_out_path, args.meta_classifier + '.model')
        with open(stacked_model_file, 'wb') as model_file:
            pickle.dump(stacked_model, model_file)

        stacked_train_preds = stacked_model.predict(train_stacked_preds)
        hard_stacked_train_preds = hard_preds(stacked_train_preds)
        stacked_train_acc = compute_accuracy(train_labels, hard_stacked_train_preds)
        print('STACKING on TRAIN', stacked_train_acc)
        out_log.write('STACKING on TRAIN {}\n'.format(stacked_train_acc))
        stacked_train_path = os.path.join(stack_out_path, 'train.scores')
        stacked_train_preds = stacked_train_preds / numpy.sum(stacked_train_preds,
                                                              axis=1,
                                                              keepdims=True)
        save_predictions(stacked_train_preds, stacked_train_path, ids=train_frame_ids)
        stacked_train_path = os.path.join(stack_out_path, 'train.submission')
        create_submission(numbers_to_letters(hard_stacked_train_preds),
                          output=stacked_train_path,
                          ids=train_frame_ids)

        stacked_valid_preds = stacked_model.predict(valid_stacked_preds)
        hard_stacked_valid_preds = hard_preds(stacked_valid_preds)
        stacked_valid_preds = stacked_valid_preds / numpy.sum(stacked_valid_preds,
                                                              axis=1,
                                                              keepdims=True)
        stacked_valid_path = os.path.join(stack_out_path, 'valid.scores')
        save_predictions(stacked_valid_preds, stacked_valid_path, ids=valid_frame_ids)
        stacked_valid_path = os.path.join(stack_out_path, 'valid.submission')
        create_submission(numbers_to_letters(hard_stacked_valid_preds),
                          output=stacked_valid_path,
                          ids=valid_frame_ids)

        #
        # AGGREGATING SINGLE EVALS ON WHOLE DATASETS
        #
        if args.aggr_func:

            aggr_out_path = os.path.join(out_path, 'aggr')
            os.makedirs(aggr_out_path, exist_ok=True)

            print('\n\nAGGREGATING WHOLE\n')
            out_log.write('\n\nAGGREGATING WHOLE\n')
            train_perfs = []

            for aggr_func in args.aggr_func:
                print('\nAggregating with', aggr_func)
                out_log.write('\nAggregating with {}\n'.format(aggr_func))

                train_aggr_preds = compute_aggregation(train_preds_list, aggr_func)

                norm_train_aggr_preds = None
                if aggr_func == 'maj':
                    hard_train_aggr_preds = train_aggr_preds
                else:
                    hard_train_aggr_preds = hard_preds(train_aggr_preds)
                    norm_train_aggr_preds = normalize_preds(
                        train_aggr_preds, cos_sim=False, eps=1e-10)
                    make_scores(norm_train_aggr_preds,
                                aggr_func,
                                aggr_out_path,
                                args.id,
                                'train',
                                train_frame_ids)

                train_acc = compute_accuracy(train_labels, hard_train_aggr_preds)

                make_submission(hard_train_aggr_preds,
                                aggr_func,
                                aggr_out_path,
                                args.id,
                                'train',
                                train_frame_ids)

                train_perfs.append((aggr_func, train_acc))

                print('Accuracy on training', train_acc)
                out_log.write('Accuracy on training {}'.format(train_acc))

                if args.valid_ext:
                    print('\nAggregating valid with', aggr_func)
                    valid_aggr_preds = compute_aggregation(test_preds_list, aggr_func)

                    norm_valid_aggr_preds = None
                    if aggr_func == 'maj':
                        hard_valid_aggr_preds = valid_aggr_preds
                    else:
                        hard_valid_aggr_preds = hard_preds(valid_aggr_preds)
                        norm_valid_aggr_preds = normalize_preds(valid_aggr_preds,
                                                                cos_sim=False, eps=1e-10)
                        make_scores(norm_valid_aggr_preds,
                                    aggr_func,
                                    aggr_out_path,
                                    args.id,
                                    'valid')

                    make_submission(hard_valid_aggr_preds,
                                    aggr_func,
                                    aggr_out_path,
                                    args.id,
                                    'valid')

            #
            # printing all train scores
            print('\n\n**** Train accuracy ranking summary ****\n')
            for aggr_func, score in reversed(sorted(train_perfs, key=lambda tup: tup[1])):
                print('{0}\t{1}'.format(aggr_func, score))

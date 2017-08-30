
from dataset import load_data_frame
from dataset import get_answers
from dataset import numbers_to_letters
from dataset import get_ids
from dataset import save_predictions
from dataset import load_tensors_preds_from_files

from evaluation import compute_accuracy
from evaluation import create_submission
from evaluation import hard_preds
from evaluation import normalize_preds

from ensemble import collect_predictions_from_dirs
from ensemble import augment_predictions
from ensemble import aggregate_function_factory
from ensemble import OneVsRestClassifier
from ensemble import OneVsOneClassifier

from embeddings import data_frame_to_matrix, data_frame_to_matrix_q_plus_a
from embeddings import load_word2vec_model
from embeddings import sum_sentence_embedding

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression

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

TMP_PATH = '/media/valerio/formalità/ranklib-tmp'

META_CLASS_DICT = {'ovr': OneVsRestClassifier,
                   'ovo': OneVsOneClassifier}


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


def create_groups(dataset, n_answers):
    n_items = dataset.shape[0]
    return numpy.array([n_answers for i in range(n_items // n_answers)])


def from_pair_matrix_to_preds(pair_matrix, n_answers):
    n_pairs = pair_matrix.shape[0]
    return pair_matrix.reshape(n_pairs // n_answers, n_answers)


def prediction_tensor_into_pairwise_matrix(preds):
    n_questions = preds.shape[0]
    n_answers = preds.shape[1]
    n_predictors = preds.shape[2]

    pred_matrix = numpy.zeros((n_questions * n_answers, n_predictors))

    for i in range(n_questions):
        for j in range(n_answers):
            for k in range(n_predictors):
                pred_matrix[i * n_answers + j, k] = preds[i, j, k]

    return pred_matrix


def vec_to_pair_vec(vec, n_answers):
    n_items = vec.shape[0]
    pair_vec = numpy.zeros((n_items, n_answers),  dtype=vec.dtype)
    for i in range(n_items):
        pair_vec[i, vec[i]] = 1

    return pair_vec.reshape(n_items * n_answers)


def pairs_to_vec(pair_vec, n_answers):
    n_pairs = pair_vec.shape[0]

    vec = pair_vec.reshape(n_pairs // n_answers, n_answers)
    return numpy.argmax(vec, axis=1)


def split_eval(train, train_labels, train_ids, perc_eval, n_answers=4):
    n_pairs = train.shape[0]
    n_items = n_pairs // n_answers
    n_train_items = int(n_items * perc_eval)
    n_pair_items = n_train_items * n_answers

    # index_items = numpy.arange(n_items)
    # index_items = numpy.random.shuffle(index_items)
    # train_index = index_items[n_pair_items:]
    # eval_index = index_items[:n_pair_items]

    return ((train[n_pair_items:], train_labels[n_pair_items:], train_ids[n_train_items:]),
            (train[:n_pair_items], train_labels[:n_pair_items], train_ids[:n_train_items]))


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

    # parser.add_argument('-e', '--eta', type=float, nargs='+',
    #                     default=[0.02],
    #                     help='Eta values')

    # parser.add_argument('-g', '--gamma', type=float, nargs='+',
    #                     default=[0.0],
    #                     help='Gamma regularizer')

    # parser.add_argument('-s', '--subsample', type=float, nargs='+',
    #                     default=[0.7],
    #                     help='Subsample percentage')

    # parser.add_argument('-k', '--col-sample-by-tree', type=float, nargs='+',
    #                     default=[0.6],
    #                     help='Col sample by tree')

    # parser.add_argument('-w', '--scale-pos-weight', type=float, nargs='+',
    #                     default=[0.8],
    #                     help='Scale pos weight')

    # parser.add_argument('-d', '--max-depth', type=int, nargs='+',
    #                     default=[8],
    #                     help='Max depth')

    # parser.add_argument('-n', '--n-rounds', type=int, nargs='+',
    #                     default=[200],
    #                     help='N rounds')

    # parser.add_argument('-l', '--n-trees', type=int, nargs='+',
    #                     default=[1],
    #                     help='N trees per round')

    # parser.add_argument('-m', '--max-delta-step', type=int, nargs='+',
    #                     default=[2],
    #                     help='Max Delta Sep')

    # parser.add_argument('-c', '--min-child-weight', type=int, nargs='+',
    #                     default=[6],
    #                     help='Min child weight')

    parser.add_argument('-c', '--meta-classifier', type=str,
                        default='ovr',
                        help='Meta classifier type (ovr|ovo)')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/l2r/mix/',
                        help='Output dir path')

    parser.add_argument('-r', '--n-rand-opt', type=int, nargs='?',
                        default=0,
                        help='Number of iterations for random grid search optimization')

    # parser.add_argument('--logify', action='store_true',
    #                     help='Considering labels to be log transformed')

    parser.add_argument('--store-models', action='store_true',
                        help='Storing models on disk as files')

    parser.add_argument('--cv', action='store_true',
                        help='Cross validating model')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-j', '--transforms', type=str, nargs='+',
                        default=['log1p'],
                        help='Transforms to apply to target var')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=0,
                        help='Verbosity level')

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
                        # default='0',
                        help='Counter to identify the output')

    parser.add_argument('--load-matrices', action='store_true',
                        help='Load numpy matrices instead of predictions')

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

    # initing the random generators
    seed = args.seed
    MAX_RAND_SEED = 99999999  # sys.maxsize
    rand_gen = random.Random(seed)
    numpy_rand_gen = numpy.random.RandomState(seed)

    n_answers = 4

    meta_classifier = META_CLASS_DICT[args.meta_classifier]
    print('Using meta classifier: {}'.format(args.meta_classifier))

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

    # train_preds = collect_predictions_from_dirs(args.preds, args.train_ext)
    # valid_preds = collect_predictions_from_dirs(args.preds, args.valid_ext)

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

    #
    # MODELS
    #
    model_names = ['xgbst1',
                   'rnknt1',
                   'lstnt1',
                   # 'lmbrnk1',
                   'lmbmrt1',
                   'rndfrt1']
    xgb_model_params = [('xgbst1', {"eta": 0.1,
                                    "min_child_weight": 10,
                                    "subsample": 0.7,
                                    "col_sample_by_tree": 0.4,
                                    "scale_pos_weight": 0.8,
                                    "max_depth": 5,
                                    "max_delta_step": 2,
                                    "n_rounds": 15,
                                    "n_trees": 50,
                                    "gamma": 0.0, })]

    ranklib_model_params = [
        ('rnknt1', {'ranker': 1,
                    'model_params': {'-epoch': 100,
                                     '-node': 20,
                                     '-lr': 0.1}}),
        ('lstnt1', {'ranker': 7,
                    'model_params': {'-epoch': 300,
                                     '-lr': 0.1}}),
        # ('lmbrnk1', {'ranker': 5,
        #              'model_params': {'-epoch': 100,
        #                               '-node': 40,
        #                               '-lr': 0.1}}),
        ('lmbmrt1', {'ranker': 6,
                     'model_params': {'-tree': 70,
                                      '-leaf': 20,
                                      '-shrinkage': 0.1,
                                      '-mls': 1}}),
        ('rndfrt1', {'ranker': 8,
                     'model_params': {'-bag': 60,
                                      '-leaf': 20,
                                      '-srate': 0.7,
                                      '-srate': 0.3}}),
    ]

    #
    # generating folds
    n_folds = args.n_folds

    cv_out_path = os.path.join(out_path, 'cv')

    # kf = KFold(train_preds.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

    perc_eval = 1 / n_folds
    t_v_s = (1 - 2 * perc_eval) / (1 - 1 / n_folds)
    print('perc eval', perc_eval)
    print('tvs', t_v_s)

    n_models = len(xgb_model_params) + len(ranklib_model_params)
    fold_train_accs = numpy.zeros((n_models, n_folds))
    fold_test_accs = numpy.zeros((n_models, n_folds))

    fold_train_preds_list = [None for i in range(n_folds)]
    fold_test_preds_list = [None for i in range(n_folds)]

    #
    # stacking
    #
    train_stacked_preds = numpy.zeros((train_preds.shape[0], n_answers, n_models))
    valid_stacked_preds = numpy.zeros((valid_preds.shape[0], n_answers, n_models, n_folds))

    #
    # making valid libsvm file
    valid_pair_matrix = prediction_tensor_into_pairwise_matrix(valid_preds)

    dataset_str = '+'.join(str(p.split('/')[-2]) for p in args.preds)

    valid_file = 'l2r_{0}.valid.pair.matrix'.format(dataset_str)
    valid_path = os.path.join(TMP_PATH, valid_file)

    #
    # and writing to it
    pair_matrix_to_ranklib_format(valid_pair_matrix, None, valid_path)

    out_log_path = os.path.join(out_path, 'exp.log')

    with open(out_log_path, 'w') as out_log:
        #
        # for each fold
        for k, (t_ids, v_ids) in enumerate(kf):

            print('\nFOLD: {0}/{1}\n'.format(k + 1, n_folds))
            # out_log.write('\nFOLD: {0}/{1}\n'.format(k + 1, n_folds))
            # out_log.flush()

            #
            # partitioning data
            train_x = prediction_tensor_into_pairwise_matrix(train_preds[t_ids])
            train_y = vec_to_pair_vec(train_labels[t_ids], n_answers)
            train_ids = train_frame_ids[t_ids]

            test_x = prediction_tensor_into_pairwise_matrix(train_preds[v_ids])
            test_y = vec_to_pair_vec(train_labels[v_ids], n_answers)
            test_ids = train_frame_ids[v_ids]

            train_labels_fold = train_labels[t_ids]
            test_labels_fold = train_labels[v_ids]

            #
            # getting a validation split for xgboost
            (train_xgb, train_ygb, train_xgb_ids),  (valid_x, valid_y, valid_ids) = \
                split_eval(train_x, train_y, train_ids, perc_eval, n_answers)

            #
            # create temporary files
            #
            # creating a temp file containing the data in ranklib format
            # param_str = '_'.join(str(k) + ':' + '>'.join(str(v) for v in params_dict[k])
            #                      for k in sorted(params_dict))

            temp_train_file = 'l2r_{0}_{1}.train.pair.matrix'.format(dataset_str, k + 1)
            temp_train_path = os.path.join(TMP_PATH, temp_train_file)

            temp_valid_file = 'l2r_{0}_{1}.valid.pair.matrix'.format(dataset_str, k + 1)
            temp_valid_path = os.path.join(TMP_PATH, temp_valid_file)

            #
            # and writing to it
            pair_matrix_to_ranklib_format(train_x, train_y, temp_train_path)
            pair_matrix_to_ranklib_format(test_x, None, temp_valid_path)

            #
            # training models
            #
            m = 0
            for i, (xgb_name, xgb_params) in enumerate(xgb_model_params):
                print('\nTraining XGB model: {0}\n{1}'.format(xgb_name,
                                                              xgb_params))
                xgb_output_path = os.path.join(cv_out_path, '{0}'.format(xgb_name))
                xgb_output_path = os.path.join(xgb_output_path, '{0}'.format(k))

                xgb_model, fold_preds, fold_perfs = train_simple_xgboost(train_xgb,
                                                                         train_ygb,
                                                                         train_xgb_ids,
                                                                         valid_x,
                                                                         valid_y,
                                                                         valid_ids,
                                                                         test_x,
                                                                         test_y,
                                                                         test_ids,
                                                                         model_name=xgb_output_path,
                                                                         silent=silent,
                                                                         **xgb_params)

                fold_train_accs[m, k] = fold_perfs[0]
                fold_test_accs[m, k] = fold_perfs[2]
                print('>>> TRAIN:{0} VALID:{1} TEST:{2} <<<\n'.format(fold_perfs[0],
                                                                      fold_perfs[1],
                                                                      fold_perfs[2]))

                #
                # TODO: predicting on all slice
                whole_train_preds, whole_train_perfs = eval_xgboost(xgb_model,
                                                                    train_x,
                                                                    train_y,
                                                                    train_ids,
                                                                    xgb_name,
                                                                    '.train.{}.scores'.format(k),
                                                                    '.train.{}.submission'.format(
                                                                        k),
                                                                    model_name=xgb_output_path)
                if fold_train_preds_list[k] is None:
                    fold_train_preds_list[k] = whole_train_preds[0]
                else:
                    fold_train_preds_list[k] = numpy.dstack((fold_train_preds_list[k],
                                                             whole_train_preds[0]))

                if fold_test_preds_list[k] is None:
                    fold_test_preds_list[k] = fold_preds[2]
                else:
                    fold_test_preds_list[k] = numpy.dstack((fold_test_preds_list[k],
                                                            fold_preds[2]))

                #
                # saving for stacking
                whole_valid_preds, whole_valid_perfs = eval_xgboost(xgb_model,
                                                                    valid_pair_matrix,
                                                                    None,
                                                                    valid_frame_ids,
                                                                    xgb_name,
                                                                    '.valid.scores',
                                                                    '.valid.submission',
                                                                    model_name=xgb_output_path)
                train_stacked_preds[v_ids, :, m] = fold_preds[2]
                valid_stacked_preds[:, :, m, k] = whole_valid_preds[0]

                m += 1

            for i, (rank_name, rank_params) in enumerate(ranklib_model_params):
                print('\nTraining ranklib model: {0}\n{1}'.format(rank_name,
                                                                  rank_params))
                rank_output_path = os.path.join(cv_out_path, '{0}'.format(rank_name))
                rank_output_path = os.path.join(rank_output_path, '{0}'.format(k))

                rank_model_path, (train_preds_fold, valid_preds_fold) = \
                    ranklib_train_eval(temp_train_path,
                                       train_ids,
                                       temp_valid_path,
                                       test_ids,
                                       train_labels_fold,
                                       output=rank_output_path,
                                       t_v_s=t_v_s,
                                       normalize=True,
                                       **rank_params)

                hard_train_preds_fold = hard_preds(train_preds_fold)
                train_perf = compute_accuracy(train_labels_fold, hard_train_preds_fold)

                hard_valid_preds_fold = hard_preds(valid_preds_fold)
                valid_perf = compute_accuracy(test_labels_fold, hard_valid_preds_fold)

                fold_train_accs[m, k] = train_perf
                fold_test_accs[m, k] = valid_perf
                print('>>> TRAIN:{0} TEST:{1} <<<\n'.format(train_perf,
                                                            valid_perf))

                if fold_train_preds_list[k] is None:
                    fold_train_preds_list[k] = train_preds_fold
                else:
                    fold_train_preds_list[k] = numpy.dstack((fold_train_preds_list[k],
                                                             train_preds_fold))

                if fold_test_preds_list[k] is None:
                    fold_test_preds_list[k] = valid_preds_fold
                else:
                    fold_test_preds_list[k] = numpy.dstack((fold_test_preds_list[k],
                                                            valid_preds_fold))

                valid_rank_output_path = os.path.join(rank_output_path,
                                                      'valid.{0}.{1}'.format(k + 1, m))
                whole_valid_preds = ranklib_eval(rank_model_path,
                                                 test_path=valid_path,
                                                 metric_2_T='P@1',
                                                 output=valid_rank_output_path)

                train_stacked_preds[v_ids, :, m] = valid_preds_fold
                valid_stacked_preds[:, :, m, k] = whole_valid_preds

                m += 1

        #
        # aggr results
        avg_train_acc = numpy.sum(fold_train_accs, axis=1) / n_folds
        avg_test_acc = numpy.sum(fold_test_accs, axis=1) / n_folds

        print('\nTRAIN ACCs')
        out_log.write('TRAIN ACCs\n')
        for i, name in enumerate(model_names):
            model_perfs_str = '{0}:\t\t{1}\t\t{2}'.format(name,
                                                          avg_train_acc[i],
                                                          '\t'.join(str(a)
                                                                    for a in fold_train_accs[i]))
            print(model_perfs_str)
            out_log.write(model_perfs_str + '\n')
            out_log.flush()

        print('VALID ACCs')
        out_log.write('VALID ACCs\n')
        for i, name in enumerate(model_names):
            model_perfs_str = '{0}:\t\t{1}\t\t{2}'.format(name,
                                                          avg_test_acc[i],
                                                          '\t'.join(str(a)
                                                                    for a in fold_test_accs[i]))
            print(model_perfs_str)
            out_log.write(model_perfs_str + '\n')
            out_log.flush()

        #
        # aggregate stacked preds for valid on folds
        valid_stacked_preds = numpy.mean(valid_stacked_preds, axis=3)
        print('VALID stacked shape', valid_stacked_preds.shape)

        if args.aggr_func:
            print('\n\nAGGREGATING\n')
            out_log.write('\n\nAGGREGATING\n')

            aggr_out_path = os.path.join(cv_out_path, 'aggr')
            for aggr_func in args.aggr_func:

                train_fold_accs = []
                test_fold_accs = []

                print('\nAggregating with', aggr_func)
                out_log.write(aggr_func + '\n')
                for k, (t_ids, v_ids) in enumerate(kf):

                    fold_test_preds = fold_test_preds_list[k]
                    test_aggr_preds = compute_aggregation(fold_test_preds, aggr_func)

                    fold_train_preds = fold_train_preds_list[k]
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
                                    aggr_out_path,
                                    args.id,
                                    'train',
                                    train_frame_ids[t_ids])
                        make_submission(hard_train_aggr_preds,
                                        aggr_func,
                                        aggr_out_path,
                                        args.id,
                                        'train',
                                        train_frame_ids[t_ids])

                        avg_train_acc = compute_accuracy(train_labels[t_ids],
                                                         hard_train_aggr_preds)
                        train_fold_accs.append(avg_train_acc)
                        print('\tTRAIN:', avg_train_acc)
                        out_log.write('\tTRAIN: {}\n'.format(avg_train_acc))

                        hard_test_aggr_preds = hard_preds(test_aggr_preds)
                        norm_test_aggr_preds = normalize_preds(test_aggr_preds,
                                                               cos_sim=False,
                                                               eps=1e-10)
                        make_scores(norm_test_aggr_preds,
                                    aggr_func,
                                    aggr_out_path,
                                    args.id,
                                    'valid',
                                    train_frame_ids[v_ids])
                        make_submission(hard_test_aggr_preds,
                                        aggr_func,
                                        aggr_out_path,
                                        args.id,
                                        'valid',
                                        train_frame_ids[v_ids])

                        avg_test_acc = compute_accuracy(train_labels[v_ids],
                                                        hard_test_aggr_preds)
                        test_fold_accs.append(avg_test_acc)
                        print('\tTEST:', avg_test_acc)
                        out_log.write('\tTEST: {}\n'.format(avg_test_acc))

                print('TRAIN folds', '\t'.join(str(f) for f in train_fold_accs))
                out_log.write('\tTRAIN folds' + '\t'.join(str(f) for f in train_fold_accs) + '\n')
                print('TEST folds', '\t'.join(str(f) for f in test_fold_accs))
                out_log.write('\tTEST folds' + '\t'.join(str(f) for f in test_fold_accs) + '\n')
                avg_train = sum(train_fold_accs) / n_folds
                avg_test = sum(test_fold_accs) / n_folds
                print('AGG TRAIN', avg_train)
                print('AGG TEST', avg_test)
                out_log.write('\tAGG TRAIN {}\n'.format(avg_train))
                out_log.write('\tAGG TEST {}\n'.format(avg_test))
                out_log.flush()

        print('\n\nSTACKING\n')
        out_log.write('\n\nSTACKING\n')
        #
        # STACKING
        #
        base_model = LogisticRegression
        base_model_params = {'fit_intercept': True,
                             'class_weight': 'balanced',
                             'penalty': 'l2',
                             'C': 10.0}

        stack_out_path = os.path.join(out_path, 'stacking')
        os.makedirs(stack_out_path, exist_ok=True)
        stack_out_train_path = os.path.join(stack_out_path, 'train.stacking')
        stack_out_valid_path = os.path.join(stack_out_path, 'valid.stacking')
        numpy.save(stack_out_train_path, train_stacked_preds)
        numpy.save(stack_out_valid_path, valid_stacked_preds)

        #
        # little cv

        accs = []
        for k, (train_ids, test_ids) in enumerate(kf):
            print('Stacking Fold', k)
            out_log.write('Stacking Fold {}\n'.format(k))

            train_x = train_stacked_preds[train_ids]
            train_y = train_labels[train_ids]
            test_x = train_stacked_preds[test_ids]
            test_y = train_labels[test_ids]

            # if calibration:
            #     caliber = Calibrator(caliber_base_model, **caliber_base_model_params)
            #     caliber.fit(train_x, train_y)
            #     train_x = caliber.predict(train_x)

            model = meta_classifier(base_model, **base_model_params)

            #
            # fitting
            print('Fitting')
            model.fit(train_x, train_y)

            #
            # predicting
            print('Predicting on train')
            train_pred_probs = model.predict(train_x)

            hard_train_preds = hard_preds(train_pred_probs)

            train_acc = compute_accuracy(train_y, hard_train_preds)
            print('ON TRAIN', train_acc)
            out_log.write('ON TRAIN {}\n'.format(train_acc))

            #
            # predicting
            print('Predicting on test')
            test_pred_probs = model.predict(test_x)

            hard_test_preds = hard_preds(test_pred_probs)

            test_acc = compute_accuracy(test_y, hard_test_preds)
            print('ON TEST', test_acc)
            out_log.write('ON TEST {}\n'.format(test_acc))
            accs.append(test_acc)

        print(accs)
        out_log.write('{}\n'.format(accs))

        print('AVG VALID', sum(accs) / n_folds)
        out_log.write('AVG VALID {}\n'.format(sum(accs) / n_folds))

        stacked_model = OneVsRestClassifier(base_model, **base_model_params)

        stacked_model.fit(train_stacked_preds, train_labels)

        stacked_train_preds = stacked_model.predict(train_stacked_preds)
        hard_stacked_train_preds = hard_preds(stacked_train_preds)
        stacked_train_acc = compute_accuracy(train_labels, hard_stacked_train_preds)
        print('STACKING on TRAIN', stacked_train_acc)
        out_log.write('STACKING on TRAIN {}\n'.format(stacked_train_acc))
        stacked_train_path = os.path.join(stack_out_path, 'train.scores')
        save_predictions(stacked_train_preds, stacked_train_path, ids=train_frame_ids)
        stacked_train_path = os.path.join(stack_out_path, 'train.submission')
        create_submission(numbers_to_letters(hard_stacked_train_preds),
                          output=stacked_train_path,
                          ids=train_frame_ids)

        stacked_valid_preds = stacked_model.predict(valid_stacked_preds)
        hard_stacked_valid_preds = hard_preds(stacked_valid_preds)
        stacked_valid_path = os.path.join(stack_out_path, 'valid.scores')
        save_predictions(stacked_valid_preds, stacked_valid_path, ids=valid_frame_ids)
        stacked_valid_path = os.path.join(stack_out_path, 'valid.submission')
        create_submission(numbers_to_letters(hard_stacked_valid_preds),
                          output=stacked_valid_path,
                          ids=valid_frame_ids)

        #
        # EVALUATING ON WHOLE SETs
        #
        print('\nLEARN+EVAL WHOLE SETs\n')
        out_log.write('\nLEARN+EVAL WHOLE SETs\n')

        train_preds_list = None
        test_preds_list = None

        #
        # partitioning data
        train_x = prediction_tensor_into_pairwise_matrix(train_preds)
        train_y = vec_to_pair_vec(train_labels, n_answers)
        train_ids = train_frame_ids

        test_x = prediction_tensor_into_pairwise_matrix(valid_preds)
        test_ids = valid_frame_ids

        #
        # getting a validation split for xgboost
        (train_xgb, train_ygb, train_xgb_ids),  (valid_x, valid_y, valid_ids) = split_eval(
            train_x, train_y, train_ids, perc_eval, n_answers)

        #
        # create temporary files
        #
        # creating a temp file containing the data in ranklib format
        # param_str = '_'.join(str(k) + ':' + '>'.join(str(v) for v in params_dict[k])
        #                      for k in sorted(params_dict))

        temp_train_file = 'l2r_{0}.train.pair.matrix'.format(dataset_str)
        temp_train_path = os.path.join(TMP_PATH, temp_train_file)

        temp_valid_file = 'l2r_{0}.valid.pair.matrix'.format(dataset_str)
        temp_valid_path = os.path.join(TMP_PATH, temp_valid_file)

        #
        # and writing to it
        pair_matrix_to_ranklib_format(train_x, train_y, temp_train_path)
        pair_matrix_to_ranklib_format(test_x, None, temp_valid_path)

        m = 0

        for i, (xgb_name, xgb_params) in enumerate(xgb_model_params):
            print('\nTraining XGB model: {0}\n{1}'.format(xgb_name,
                                                          xgb_params))
            xgb_output_path = os.path.join(out_path, '{0}'.format(xgb_name))

            xgb_model, fold_preds, fold_perfs = train_simple_xgboost(train_xgb,
                                                                     train_ygb,
                                                                     train_xgb_ids,
                                                                     valid_x,
                                                                     valid_y,
                                                                     valid_ids,
                                                                     None,
                                                                     None,
                                                                     None,
                                                                     model_name=xgb_output_path,
                                                                     silent=silent,
                                                                     **xgb_params)

            fold_train_accs[m, k] = fold_perfs[0]
            fold_test_accs[m, k] = fold_perfs[2]
            print('>>> TRAIN:{0} VALID:{1} TEST:{2} <<<'.format(fold_perfs[0],
                                                                fold_perfs[1],
                                                                fold_perfs[2]))
            out_log.write('>>> TRAIN:{0} VALID:{1} TEST:{2} <<<\n'.format(fold_perfs[0],
                                                                          fold_perfs[1],
                                                                          fold_perfs[2]))
            #
            # TODO: predicting on all slice
            whole_train_preds, whole_train_perfs = eval_xgboost(xgb_model,
                                                                train_x,
                                                                train_y,
                                                                train_ids,
                                                                xgb_name,
                                                                '.train.scores',
                                                                '.train.submission',
                                                                model_name=xgb_output_path)

            whole_test_preds, whole_test_perfs = eval_xgboost(xgb_model,
                                                              test_x,
                                                              None,
                                                              test_ids,
                                                              xgb_name,
                                                              '.valid.scores',
                                                              '.valid.submission',
                                                              model_name=xgb_output_path)

            if train_preds_list is None:
                train_preds_list = whole_train_preds[0]
            else:
                train_preds_list = numpy.dstack((train_preds_list,
                                                 whole_train_preds[0]))

            if test_preds_list is None:
                test_preds_list = whole_test_preds[0]
            else:
                test_preds_list = numpy.dstack((test_preds_list,
                                                whole_test_preds[0]))

            m += 1

        for i, (rank_name, rank_params) in enumerate(ranklib_model_params):
            print('\nTraining ranklib model: {0}\n{1}'.format(rank_name,
                                                              rank_params))
            out_log.write('\nTraining ranklib model: {0}\n{1}\n'.format(rank_name,
                                                                        rank_params))
            rank_output_path = os.path.join(out_path, '{0}'.format(rank_name))

            rank_model_path, (train_preds_eval, valid_preds_eval) = \
                ranklib_train_eval(temp_train_path,
                                   train_ids,
                                   temp_valid_path,
                                   test_ids,
                                   train_labels,
                                   output=rank_output_path,
                                   t_v_s=t_v_s,
                                   normalize=True,
                                   **rank_params)

            hard_train_preds_fold = hard_preds(train_preds_fold)
            train_perf = compute_accuracy(train_labels_fold, hard_train_preds_fold)

            hard_valid_preds_fold = hard_preds(valid_preds_fold)

            print('>>> TRAIN:{0} <<<'.format(train_perf))
            out_log.write('>>> TRAIN:{0} <<<\n'.format(train_perf))

            if train_preds_list is None:
                train_preds_list = train_preds_eval
            else:
                train_preds_list = numpy.dstack((train_preds_list,
                                                 train_preds_eval))

            if test_preds_list is None:
                test_preds_list = valid_preds_eval
            else:
                test_preds_list = numpy.dstack((test_preds_list,
                                                valid_preds_eval))

            m += 1

        if args.aggr_func:

            aggr_out_path = os.path.join(out_path, 'aggr')
            print('\nAGGREGATING\n')
            out_log.write('\nAGGREGATING\n')

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
                                train_ids)

                train_acc = compute_accuracy(train_labels, hard_train_aggr_preds)

                make_submission(hard_train_aggr_preds,
                                aggr_func,
                                aggr_out_path,
                                args.id,
                                'train',
                                train_ids)

                train_perfs.append((aggr_func, train_acc))

                print('Accuracy on training', train_acc)
                out_log.write('Accuracy on training {}\n'.format(train_acc))

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
            out_log.write('\n\n**** Train accuracy ranking summary ****\n')
            for aggr_func, score in reversed(sorted(train_perfs, key=lambda tup: tup[1])):
                print('{0}\t{1}'.format(aggr_func, score))
                out_log.write('{0}\t{1}'.format(aggr_func, score))


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

from ensemble import collect_predictions_from_dirs
from ensemble import predictions_paths_from_dirs

from ensemble import augment_predictions
from ensemble import OneVsRestClassifier
from ensemble import OneVsOneClassifier
from ensemble import feature_importances_meta_clf
from ensemble import feature_importances_meta_clf_cv

from embeddings import data_frame_to_matrix, data_frame_to_matrix_q_plus_a
from embeddings import load_word2vec_model
from embeddings import sum_sentence_embedding

from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import pickle

import numpy

import random

import datetime

import itertools

import os

import logging

import xgboost as xgb


TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'


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

MODEL_EXT = '.xgb.model'
EPS = 1e-10


def eval_xgboost(model,
                 dataset,
                 labels,
                 ids,
                 prefix_str,
                 score_ext,
                 submission_ext,
                 metric=compute_accuracy,
                 model_name=None,
                 n_answers=4,
                 early_rounds=300):

    # print(dataset.shape)
    # print(ids.shape)
    if labels is not None:
        print(labels.shape)

    xgb_matrix = xgb.DMatrix(dataset, label=labels)
    groups = create_groups(dataset, n_answers)
    xgb_matrix.set_group(groups)

    if early_rounds > 0:
        preds = model.predict(xgb_matrix, ntree_limit=model.best_ntree_limit)
    else:
        preds = model.predict(xgb_matrix)

    preds = from_pair_matrix_to_preds(preds, n_answers)
    preds = normalize_preds(preds, cos_sim=False, eps=EPS)

    h_preds = hard_preds(preds)
    # print(preds.shape, h_preds.shape, len(ids))

    perf = 0.0
    if labels is not None:
        #
        # measuring performances
        labels_vec = pairs_to_vec(labels, n_answers)
        perf = metric(labels_vec, h_preds)
        print('Eval', perf)

    if model_name and ids is not None:

        os.makedirs(model_name, exist_ok=True)

        pred_file = os.path.join(model_name, prefix_str + score_ext)
        save_predictions(preds, pred_file, ids=ids)

        pred_file = os.path.join(model_name, prefix_str + submission_ext)
        create_submission(numbers_to_letters(h_preds),
                          output=pred_file,
                          ids=ids)

    return (preds, h_preds), perf


def train_simple_xgboost(train,
                         train_labels,
                         train_ids,
                         valid,
                         valid_labels,
                         valid_ids,
                         test,
                         test_labels,
                         test_ids,
                         eta,
                         min_child_weight,
                         subsample,
                         col_sample_by_tree,
                         scale_pos_weight,
                         max_depth,
                         max_delta_step,
                         n_rounds,
                         n_trees,
                         silent,
                         gamma=0.0,
                         metric=compute_accuracy,
                         model_name=None,
                         n_answers=4,
                         early_rounds=300,
                         save_model=False):
    """
    A single run of xgboost
    """

    if valid is not None:
        assert train.shape[1] == valid.shape[1]

    #
    # setting parameters
    xgb_params = {}
    xgb_params["objective"] = "rank:pairwise"
    xgb_params["eval_metric"] = "map@1"
    xgb_params["eta"] = eta
    xgb_params["min_child_weight"] = min_child_weight
    xgb_params["subsample"] = subsample
    xgb_params["colsample_bytree"] = col_sample_by_tree
    xgb_params["scale_pos_weight"] = scale_pos_weight
    xgb_params["silent"] = silent
    xgb_params["num_parallel_tree"] = n_trees
    xgb_params["max_depth"] = max_depth
    xgb_params["max_delta_step"] = max_delta_step
    xgb_params["gamma"] = gamma

    prefix_str = 'xgb_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}'.format(eta,
                                                              min_child_weight,
                                                              subsample,
                                                              col_sample_by_tree,
                                                              scale_pos_weight,
                                                              max_depth,
                                                              max_delta_step,
                                                              gamma)

    param_list = list(xgb_params.items())

    #
    # creating matrices
    xgb_train = xgb.DMatrix(train, label=train_labels)
    train_groups = create_groups(train, n_answers)
    xgb_train.set_group(train_groups)

    xgb_valid = None
    if valid is not None:
        xgb_valid = xgb.DMatrix(valid, label=valid_labels)
        valid_groups = create_groups(valid, n_answers)
        xgb_valid.set_group(valid_groups)

    xgb_test = None
    if test is not None:
        xgb_test = xgb.DMatrix(test, label=test_labels)
        test_group = create_groups(test, n_answers)
        xgb_test.set_group(test_group)

    #
    # create a watchlist
    if test_labels is not None:
        watchlist = [(xgb_train, 'train'), (xgb_test, 'eval'), (xgb_valid, 'eval')]
    else:
        watchlist = [(xgb_train, 'train'), (xgb_valid, 'eval')]

    #
    # training
    model = xgb.train(param_list, xgb_train, n_rounds, watchlist,
                      early_stopping_rounds=early_rounds)

    #
    # predicting
    if early_rounds > 0:
        train_preds = model.predict(xgb_train, ntree_limit=model.best_ntree_limit)
    else:
        train_preds = model.predict(xgb_train)

    valid_preds = None
    if valid is not None:
        if early_rounds > 0:
            valid_preds = model.predict(xgb_valid, ntree_limit=model.best_ntree_limit)
        else:
            valid_preds = model.predict(xgb_valid)

    test_preds = None
    if test is not None:
        if early_rounds > 0:
            test_preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
        else:
            test_preds = model.predict(xgb_test)

    #
    # transforming them back
    train_preds = from_pair_matrix_to_preds(train_preds, n_answers)
    train_preds = normalize_preds(train_preds, cos_sim=False, eps=EPS)

    if valid_preds is not None:
        valid_preds = from_pair_matrix_to_preds(valid_preds, n_answers)
        valid_preds = normalize_preds(valid_preds, cos_sim=False, eps=EPS)
    if test_preds is not None:
        test_preds = from_pair_matrix_to_preds(test_preds, n_answers)
        test_preds = normalize_preds(test_preds, cos_sim=False, eps=EPS)

    #
    # hard preds for accuracy
    hard_train_preds = hard_preds(train_preds)
    if valid_preds is not None:
        hard_valid_preds = hard_preds(valid_preds)
    if test_preds is not None:
        hard_test_preds = hard_preds(test_preds)

    #
    # measuring performances
    train_labels_vec = pairs_to_vec(train_labels, n_answers)
    # print(train_labels_vec.shape, hard_train_preds.shape)
    train_perf = metric(train_labels_vec, hard_train_preds)
    valid_perf = 0.0
    test_perf = 0.0

    if valid is not None:
        valid_labels_vec = pairs_to_vec(valid_labels, n_answers)
        valid_perf = metric(valid_labels_vec, hard_valid_preds)

    if test_labels is not None:
        test_labels_vec = pairs_to_vec(test_labels, n_answers)
        test_perf = metric(test_labels_vec, hard_test_preds)

    #
    # saving model
    if model_name:

        os.makedirs(model_name, exist_ok=True)

        if save_model:
            model_fullname = model_name + MODEL_EXT
            save_python = True

            if save_python:
                with open(model_fullname, 'wb') as model_file:
                    pickle.dump(model, model_file)
            else:
                # model_featuremap = model_name + FEATURE_MAP_EXT
                model.save_model(model_fullname)

        train_pred_file = os.path.join(model_name, prefix_str + '.t.scores')
        save_predictions(train_preds, train_pred_file, ids=train_ids)

        if train_ids is not None:
            train_pred_file = os.path.join(model_name, prefix_str + 't.submission')
            create_submission(numbers_to_letters(hard_train_preds),
                              output=train_pred_file,
                              ids=train_ids)

        if valid is not None:
            valid_pred_file = os.path.join(model_name, prefix_str + 'v.scores')
            save_predictions(valid_preds, valid_pred_file, ids=valid_ids)

            if valid_ids is not None:
                valid_pred_file = os.path.join(model_name, prefix_str + 'v.submission')
                create_submission(numbers_to_letters(hard_valid_preds),
                                  output=valid_pred_file,
                                  ids=valid_ids)

        if test is not None:
            test_pred_file = os.path.join(model_name, prefix_str + 'vv.scores')
            save_predictions(test_preds, test_pred_file, ids=test_ids)

            if test_ids is not None:
                test_pred_file = os.path.join(model_name, prefix_str + 'vv.submission')
                create_submission(numbers_to_letters(hard_test_preds),
                                  output=test_pred_file,
                                  ids=test_ids)

    return model, (train_preds, valid_preds, test_preds), (train_perf, valid_perf, test_perf)


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


def cv_train_simple_xgboost(folds,
                            eta,
                            min_child_weight,
                            subsample,
                            col_sample_by_tree,
                            scale_pos_weight,
                            max_depth,
                            max_delta_step,
                            n_rounds,
                            n_trees,
                            silent,
                            gamma,
                            metric=compute_accuracy,
                            model_name=None,
                            n_answers=4,
                            perc_eval=None):

    train_perfs = []
    valid_perfs = []
    # models = []

    n_folds = len(folds)

    if not perc_eval:
        perc_eval = 1 / n_folds

    for i, ((train, train_labels, train_ids),
            (valid, valid_labels, valid_ids)) in enumerate(folds):

        # print('cv train', train.shape, train_labels.shape, train_ids.shape)
        # print('cv valid', valid.shape, valid_labels.shape, valid_ids.shape)
        print('Executing fold {0}/{1}'.format(i + 1, n_folds))
        fold_model_name = None
        if model_name:
            fold_model_name = model_name + '/{0}/.'.format(i)

            os.makedirs(fold_model_name, exist_ok=True)

        (train, train_labels, train_ids),  (eval_set, eval_labels, eval_ids) = \
            split_eval(train, train_labels, train_ids, perc_eval, n_answers)
        #
        # training and evaluating for the fold
        fold_model, fold_preds, fold_perfs = \
            train_simple_xgboost(train,
                                 train_labels,
                                 train_ids,
                                 eval_set,
                                 eval_labels,
                                 eval_ids,
                                 valid,
                                 valid_labels,
                                 valid_ids,
                                 eta=eta,
                                 min_child_weight=min_child_weight,
                                 subsample=subsample,
                                 col_sample_by_tree=col_sample_by_tree,
                                 scale_pos_weight=scale_pos_weight,
                                 max_depth=max_depth,
                                 max_delta_step=max_delta_step,
                                 n_rounds=n_rounds,
                                 n_trees=n_trees,
                                 silent=silent,
                                 gamma=gamma,
                                 metric=compute_accuracy,
                                 model_name=fold_model_name,
                                 n_answers=n_answers)

        print('**\n', fold_perfs, '\n**\n')
        train_perfs.append(fold_perfs[0])
        valid_perfs.append(fold_perfs[2])  # not 1 anymore

    return (numpy.sum(train_perfs) / n_folds,
            numpy.sum(valid_perfs) / n_folds), (train_perfs, valid_perfs)


def xgboost_grid_search(train,
                        train_labels,
                        train_ids,
                        valid,
                        valid_labels,
                        valid_ids,
                        test,
                        test_ids,
                        test_labels,
                        etas,
                        min_child_weights,
                        subsamples,
                        col_sample_by_trees,
                        scale_pos_weights,
                        max_depths,
                        max_delta_steps,
                        n_rounds,
                        n_trees,
                        n_rand_opt,
                        verbose,
                        gammas,
                        perf_metric,
                        out_path,
                        save_models=False):
    """
    Grid search
    """

    out_log_path = os.path.join(out_path, '/exp.log')
    # grid_models_path = None
    # if save_models:
    #     grid_models_path = out_path + '/grid-models'
    #     if not os.path.exists(os.path.dirname(grid_models_path)):
    #         os.makedirs(os.path.dirname(grid_models_path))

    best_valid_acc = 0.0

    best_state = {}
    best_model = None
    # to keep other stuff than model parameters
    best_stats_dict = {}

    with open(out_log_path, 'w') as out_log:

        preamble = ("""id:\teta:\tmin_child_weight:\tsubsample:\tcolsample:\tscale:""" +
                    """\tdepth:\tdelta:\tgamma:\trounds:""" +
                    """\ttrain\tvalid:\ttest\n""")

        silent = 1 if verbose < 1 else 0

        # out_log.write("parameters:\n{0}\n\n".format(args))
        print(preamble)
        out_log.write(preamble)
        out_log.flush()

        if n_rand_opt > 0:
            logging.info('Random search')

        else:
            possible_configurations = itertools.product(etas,
                                                        min_child_weights,
                                                        subsamples,
                                                        col_sample_by_trees,
                                                        scale_pos_weights,
                                                        max_depths,
                                                        max_delta_steps,
                                                        gammas,
                                                        n_rounds,
                                                        n_trees)

        #
        # shall we transform the dependent var?
        # label_transforms = {}
        # if logify:
        #     label_transforms['label'] = [numpy.log1p, numpy.expm1]

        model_id = 0

        for eta, min_child_weight, subsample, colsample,\
                scale, depth, delta, gamma, rounds, n_tree in possible_configurations:

            config_string = ('id:{8} eta:{0} min-child-weight:{1}' +
                             'subsample:{2} colsample:{3} scale:{4} depth:{5} delta:{6}' +
                             ' rounds:{7}').format(eta, min_child_weight, subsample, colsample,
                                                   scale, depth, delta, rounds, model_id)
            logging.info('Checking configuration: %s', config_string)

            #
            # creating a model name file, if necessary

            model_name = None
            if save_models:
                model_name = os.path.join(out_path, str(model_id),  str(model_id))
                if not os.path.exists(os.path.dirname(model_name)):
                    os.makedirs(os.path.dirname(model_name))

            #
            # calling the single iteration
            model, predictions, performances = train_simple_xgboost(train,
                                                                    train_labels,
                                                                    train_ids,
                                                                    valid,
                                                                    valid_labels,
                                                                    valid_ids,
                                                                    test,
                                                                    test_ids,
                                                                    test_labels,
                                                                    eta,
                                                                    min_child_weight,
                                                                    subsample,
                                                                    colsample,
                                                                    scale,
                                                                    depth,
                                                                    delta,
                                                                    rounds,
                                                                    n_tree,
                                                                    silent,
                                                                    gamma,
                                                                    perf_metric,
                                                                    model_name=model_name)

            train_perf = performances[0]
            valid_perf = performances[1]

            test_perf = 0.0
            # if test is not None:

            #     test_rmsle = compute_rmsle(test_labels,
            #                                test_preds)

            #
            # saving best values
            if valid_perf > best_valid_acc:
                best_valid_acc = valid_perf
                best_model = model
                best_state["objective"] = "rank:pairwise"
                best_state["silent"] = silent
                best_state["eta"] = eta
                best_state["min_child_weight"] = min_child_weight
                best_state["subsample"] = subsample
                best_state["colsample_bytree"] = colsample
                best_state["scale_pos_weight"] = scale
                best_state["max_depth"] = depth
                best_state["max_delta_step"] = delta
                best_state['rounds'] = rounds
                best_state['gamma'] = gamma
                best_stats_dict['id'] = model_id

            #
            # writing to file a line for the grid
            stats = '\t'.join([str(c) for c in [model_id, eta,
                                                min_child_weight,
                                                subsample,
                                                colsample,
                                                scale, depth,
                                                delta,
                                                gamma,
                                                rounds,
                                                train_perf,
                                                valid_perf,
                                                test_perf]])
            print(stats)
            out_log.write(stats + '\n')
            out_log.flush()

            #
            # incrementing id
            model_id += 1

        #
        # adding more options but just on the print
        p_best_state = dict(best_state)
        p_best_state.update(best_stats_dict)
        logging.info('Grid search ended, best state')
        out_log.write("{0}".format(p_best_state))
        out_log.flush()

        print('Best params')
        print(p_best_state)
        print(best_state)

        return best_model, best_state


def cv_xgboost_grid_search(folds,
                           etas,
                           min_child_weights,
                           subsamples,
                           col_sample_by_trees,
                           scale_pos_weights,
                           max_depths,
                           max_delta_steps,
                           n_rounds,
                           n_trees,
                           n_rand_opt,
                           verbose,
                           gammas,
                           perf_metric,
                           out_path,
                           save_models=False):
    """
    Grid search
    """

    os.makedirs(out_path, exist_ok=True)
    out_log_path = os.path.join(out_path, 'exp.log')

    best_valid_acc = 0.0

    best_state = {}
    best_model = None
    # to keep other stuff than model parameters
    best_stats_dict = {}

    with open(out_log_path, 'w') as out_log:

        folds_str = '\t'.join('fold_{}'.format(i) for i in range(n_folds))
        preamble = ("""id:\teta:\tmin_child_weight:\tsubsample:\tcolsample:\tscale:""" +
                    """\tdepth:\tdelta:\tgamma:\trounds:\ttrees:\ttrain:\tvalid:""" +
                    """\t{}:\n""".format(folds_str))

        silent = 1 if verbose < 1 else 0

        # out_log.write("parameters:\n{0}\n\n".format(args))
        print(preamble)
        out_log.write(preamble)
        out_log.flush()

        if n_rand_opt > 0:
            logging.info('Random search')

        else:
            possible_configurations = itertools.product(etas,
                                                        min_child_weights,
                                                        subsamples,
                                                        col_sample_by_trees,
                                                        scale_pos_weights,
                                                        max_depths,
                                                        max_delta_steps,
                                                        gammas,
                                                        n_rounds,
                                                        n_trees)

        #
        # shall we transform the dependent var?
        # label_transforms = {}
        # if logify:
        #     label_transforms['label'] = [numpy.log1p, numpy.expm1]

        model_id = 0

        for eta, min_child_weight, subsample, colsample,\
                scale, depth, delta, gamma, rounds, trees in possible_configurations:

            config_string = ('id:{8} eta:{0} min-child-weight:{1}' +
                             'subsample:{2} colsample:{3} scale:{4} depth:{5} delta:{6}' +
                             ' rounds:{7} trees:{9}').format(eta, min_child_weight, subsample, colsample,
                                                             scale, depth, delta, rounds,  model_id, trees)
            logging.info('Checking configuration: %s', config_string)

            #
            # creating a model name file, if necessary

            model_name = None
            if save_models:
                model_name = os.path.join(out_path, str(model_id))
                if not os.path.exists(os.path.dirname(model_name)):
                    os.makedirs(os.path.dirname(model_name))

            #
            # calling the single iteration
            (train_perf, valid_perf), history_perfs = \
                cv_train_simple_xgboost(folds,
                                        eta,
                                        min_child_weight,
                                        subsample,
                                        colsample,
                                        scale,
                                        depth,
                                        delta,
                                        rounds,
                                        trees,
                                        silent,
                                        gamma,
                                        perf_metric,
                                        model_name=model_name)

            # if test is not None:

            #     test_rmsle = compute_rmsle(test_labels,
            #                                test_preds)

            #
            # saving best values
            if valid_perf > best_valid_acc:
                best_valid_acc = valid_perf
                best_state["objective"] = "rank:pairwise"
                best_state["silent"] = silent
                best_state["eta"] = eta
                best_state["min_child_weight"] = min_child_weight
                best_state["subsample"] = subsample
                best_state["colsample_bytree"] = colsample
                best_state["scale_pos_weight"] = scale
                best_state["max_depth"] = depth
                best_state["max_delta_step"] = delta
                best_state['rounds'] = rounds
                best_state['gamma'] = gamma
                best_state['num_parallel_tree'] = trees
                best_stats_dict['id'] = model_id

            #
            # writing to file a line for the grid
            perf_str = '\t'.join([str(c) for c in history_perfs[1]])
            stats = '\t'.join([str(c) for c in [model_id, eta,
                                                min_child_weight,
                                                subsample,
                                                colsample,
                                                scale, depth,
                                                delta,
                                                gamma,
                                                rounds,
                                                trees,
                                                train_perf,
                                                valid_perf,
                                                perf_str]])
            print(stats)
            out_log.write(stats + '\n')
            out_log.flush()

            #
            # incrementing id
            model_id += 1

        #
        # adding more options but just on the print
        p_best_state = dict(best_state)
        p_best_state.update(best_stats_dict)
        logging.info('Grid search ended, best state')
        out_log.write("{0}".format(p_best_state))
        out_log.flush()

        print('Best params')
        print(p_best_state)
        print(best_state)

        return best_model, best_state


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

    parser.add_argument('-e', '--eta', type=float, nargs='+',
                        default=[0.02],
                        help='Eta values')

    parser.add_argument('-g', '--gamma', type=float, nargs='+',
                        default=[0.0],
                        help='Gamma regularizer')

    parser.add_argument('-s', '--subsample', type=float, nargs='+',
                        default=[0.7],
                        help='Subsample percentage')

    parser.add_argument('-k', '--col-sample-by-tree', type=float, nargs='+',
                        default=[0.6],
                        help='Col sample by tree')

    parser.add_argument('-w', '--scale-pos-weight', type=float, nargs='+',
                        default=[0.8],
                        help='Scale pos weight')

    parser.add_argument('-d', '--max-depth', type=int, nargs='+',
                        default=[8],
                        help='Max depth')

    parser.add_argument('-n', '--n-rounds', type=int, nargs='+',
                        default=[200],
                        help='N rounds')

    parser.add_argument('-l', '--n-trees', type=int, nargs='+',
                        default=[1],
                        help='N trees per round')

    parser.add_argument('-m', '--max-delta-step', type=int, nargs='+',
                        default=[2],
                        help='Max Delta Sep')

    parser.add_argument('-c', '--min-child-weight', type=int, nargs='+',
                        default=[6],
                        help='Min child weight')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/xgboost/',
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

    parser.add_argument('--train-ext', type=str,
                        default=TRAIN_PREDS_EXT,
                        help='Train preds file extension')

    parser.add_argument('--valid-ext', type=str,
                        default=VALID_PREDS_EXT,
                        help='Valid preds file extension')
    parser.add_argument('--load-matrices', action='store_true',
                        help='Load numpy matrices instead of predictions')

    parser.add_argument('--feature-selection', type=float, nargs='?',
                        default=None,
                        help='Feature selection threshold to apply to feature importances')
    parser.add_argument('--aug-features', type=str, nargs='+',
                        help='Augment features by passing feature map files paths')

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
    numpy.random.seed(seed)

    n_answers = 4

    #
    # loading frames, labels, ids
    train_frame = load_data_frame(args.train)
    valid_frame = load_data_frame(args.test)

    train_labels = numpy.array(get_answers(train_frame, numeric=True))
    # train_labels = vec_to_pair_vec(train_labels_vec, n_answers)

    train_ids = numpy.array(get_ids(train_frame))
    valid_ids = numpy.array(get_ids(valid_frame))

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
    # converting them to pairwise matrices
    train_matrix = prediction_tensor_into_pairwise_matrix(train_preds)
    valid_matrix = prediction_tensor_into_pairwise_matrix(valid_preds)

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

    print(out_path)

    # #
    # # creating dir if non-existant
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if args.cv:

        #
        # generating folds
        n_folds = args.n_folds

        # kf = KFold(train_preds.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
        kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

        word2_vec_aug = False

        if word2_vec_aug:
            w2v_model = load_word2vec_model('../models/word2vec/065/0/0.model')
            folds = []
            for t_ids, v_ids in kf:
                train_pair_matrix = prediction_tensor_into_pairwise_matrix(train_preds[t_ids])
                w2v_train_matrix, t_labels = data_frame_to_matrix_q_plus_a(train_frame,
                                                                           w2v_model,
                                                                           sum_sentence_embedding,
                                                                           ids=t_ids)

                valid_pair_matrix = prediction_tensor_into_pairwise_matrix(train_preds[v_ids])
                w2v_valid_matrix, v_labels = data_frame_to_matrix_q_plus_a(train_frame,
                                                                           w2v_model,
                                                                           sum_sentence_embedding,
                                                                           ids=v_ids)
                print(train_pair_matrix.shape, w2v_train_matrix.shape)
                print(valid_pair_matrix.shape, w2v_valid_matrix.shape)
                folds.append(((numpy.hstack((train_pair_matrix, w2v_train_matrix)),
                               vec_to_pair_vec(train_labels[t_ids], n_answers),
                               train_ids[t_ids]),
                              (numpy.hstack((valid_pair_matrix, w2v_valid_matrix)),
                               vec_to_pair_vec(train_labels[v_ids], n_answers),
                               train_ids[v_ids])))

        else:
            folds = [((prediction_tensor_into_pairwise_matrix(train_preds[t_ids]),
                       vec_to_pair_vec(train_labels[t_ids], n_answers),
                       train_ids[t_ids]),
                      (prediction_tensor_into_pairwise_matrix(train_preds[v_ids]),
                       vec_to_pair_vec(train_labels[v_ids], n_answers),
                       train_ids[v_ids]))
                     for t_ids, v_ids in kf]

        #
        # running the cross validation for param selection
        best_model, best_params = cv_xgboost_grid_search(folds,
                                                         etas=args.eta,
                                                         min_child_weights=args.min_child_weight,
                                                         subsamples=args.subsample,
                                                         col_sample_by_trees=args.col_sample_by_tree,
                                                         scale_pos_weights=args.scale_pos_weight,
                                                         max_depths=args.max_depth,
                                                         max_delta_steps=args.max_delta_step,
                                                         n_rounds=args.n_rounds,
                                                         n_trees=args.n_trees,
                                                         n_rand_opt=args.n_rand_opt,
                                                         verbose=args.verbose,
                                                         gammas=args.gamma,
                                                         perf_metric=compute_accuracy,
                                                         out_path=out_path,
                                                         save_models=args.store_models)

        #
        # training on the whole training and validating
        if word2_vec_aug:
            train_pair_matrix = prediction_tensor_into_pairwise_matrix(train_preds)
            w2v_train_matrix, t_labels = data_frame_to_matrix_q_plus_a(train_frame,
                                                                       w2v_model,
                                                                       sum_sentence_embedding,
                                                                       labels=False)

            valid_pair_matrix = prediction_tensor_into_pairwise_matrix(valid_preds)
            w2v_valid_matrix, v_labels = data_frame_to_matrix_q_plus_a(valid_frame,
                                                                       w2v_model,
                                                                       sum_sentence_embedding,
                                                                       labels=False)
            model, preds, (train_acc, eval_acc,  valid_acc) = \
                train_simple_xgboost(numpy.hstack((train_pair_matrix,
                                                   w2v_train_matrix)),
                                     vec_to_pair_vec(train_labels, n_answers),
                                     train_ids,
                                     None, None, None,
                                     numpy.hstack((valid_pair_matrix,
                                                   w2v_valid_matrix)),
                                     None,
                                     valid_ids,
                                     eta=best_params['eta'],
                                     min_child_weight=best_params['min_child_weight'],
                                     subsample=best_params['subsample'],
                                     col_sample_by_tree=best_params['colsample_bytree'],
                                     scale_pos_weight=best_params['scale_pos_weight'],
                                     max_depth=best_params['max_depth'],
                                     max_delta_step=best_params['max_delta_step'],
                                     n_rounds=best_params['rounds'],
                                     n_trees=best_params['num_parallel_tree'],
                                     silent=1,
                                     model_name=None)
            os.path.join(out_path, 'l2r')
            print(train_acc, valid_acc)

        else:
            #
            # shuffling
            n_questions = train_preds.shape[0]
            print(train_preds.shape, train_labels.shape)
            train_index = numpy.arange(n_questions)
            numpy.random.shuffle(train_index)
            print(train_index.shape, train_index)
            train_preds_shuffled, train_labels_shuffled, train_ids_shuffled = \
                train_preds[train_index, :, :], train_labels[train_index], train_ids[train_index]
            print(train_preds_shuffled.shape, train_labels_shuffled.shape, train_ids_shuffled)

            #
            #
            train = prediction_tensor_into_pairwise_matrix(train_preds_shuffled)
            train_labels_vec = vec_to_pair_vec(train_labels_shuffled, n_answers)
            print('Before removing eval', train.shape, train_labels_vec.shape)
            perc_eval = 0.1

            (train, train_labels_vec, train_ids_split),  (eval_set, eval_labels_vec, eval_ids) = \
                split_eval(train, train_labels_vec, train_ids_shuffled, perc_eval, n_answers)

            print('After removing eval', train.shape, train_labels_vec.shape, train_ids.shape,
                  eval_set.shape, eval_labels_vec.shape, eval_ids.shape)

            model_name = os.path.join(out_path, 'l2r')

            model, preds, (train_acc, eval_acc, valid_acc) = \
                train_simple_xgboost(train,
                                     train_labels_vec,
                                     train_ids_split,
                                     eval_set, eval_labels_vec, eval_ids,
                                     prediction_tensor_into_pairwise_matrix(valid_preds),
                                     None,
                                     valid_ids,
                                     eta=best_params['eta'],
                                     min_child_weight=best_params['min_child_weight'],
                                     subsample=best_params['subsample'],
                                     col_sample_by_tree=best_params['colsample_bytree'],
                                     scale_pos_weight=best_params['scale_pos_weight'],
                                     max_depth=best_params['max_depth'],
                                     max_delta_step=best_params['max_delta_step'],
                                     n_rounds=best_params['rounds'],  # * 2,
                                     n_trees=best_params['num_parallel_tree'],
                                     silent=1,
                                     gamma=best_params['gamma'],
                                     model_name=model_name,
                                     save_model=True)
            print(train_acc, eval_acc, valid_acc)
            #
            # now evaluating on everything

            train = prediction_tensor_into_pairwise_matrix(train_preds)
            train_labels_vec = vec_to_pair_vec(train_labels, n_answers)
            print('All data', train.shape, train_labels_vec.shape, len(train_ids))

            prefix_str = ('xgb_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}'.
                          format(best_params['eta'],
                                 best_params['min_child_weight'],
                                 best_params['subsample'],
                                 best_params['colsample_bytree'],
                                 best_params['scale_pos_weight'],
                                 best_params['max_depth'],
                                 best_params['max_delta_step'],
                                 best_params['gamma'],
                                 best_params['rounds'],
                                 best_params['num_parallel_tree']))
            #
            # eval on train
            eval_preds, eval_perfs = eval_xgboost(model,
                                                  train,
                                                  train_labels_vec,
                                                  train_ids,
                                                  prefix_str,
                                                  '.train.scores',
                                                  '.train.submission',
                                                  metric=compute_accuracy,
                                                  model_name=model_name,
                                                  n_answers=4,
                                                  early_rounds=300)
            #
            # eval on valid
            eval_preds, eval_perfs = eval_xgboost(model,
                                                  valid_matrix,
                                                  None,
                                                  valid_ids,
                                                  prefix_str,
                                                  '.valid.scores',
                                                  '.valid.submission',
                                                  metric=compute_accuracy,
                                                  model_name=model_name,
                                                  n_answers=4,
                                                  early_rounds=300)

    else:
        raise ValueError('Only cv implementemented for now!')

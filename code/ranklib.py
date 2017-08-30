import numpy

import pandas

import os

import re

import subprocess

import itertools

from dataset import load_data_frame
from dataset import get_ids
from dataset import get_answers
from dataset import save_predictions
from dataset import numbers_to_letters

from ensemble import collect_predictions_from_dirs
from ensemble import augment_predictions

from evaluation import normalize_preds
from evaluation import hard_preds
from evaluation import compute_accuracy
from evaluation import create_submission

from time import perf_counter

import argparse

import logging
import datetime

from sklearn.cross_validation import StratifiedKFold

TRAIN_SUBS_EXT = 'train.submission'
VALID_SUBS_EXT = 'valid.submission'

TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'

OUT_PREDS_EXT = '.preds'

TMP_PATH = '/media/valerio/formalità/ranklib-tmp'

# java -jar RankLib-2.1-patched.jar -train '/home/valerio/Petto
# Redigi/kaggle/allen-ai-science-qa/data/80_preds_liblinear_format'
# -ranker 4 -metric2t P@1 -metric2T P@1 -gmax 1 -norm sum -kcv 10  -reg
# 0.001
JAVA_CMD_PATH = 'java'
JAVA_CMD_OPTS = '-jar'
RANKLIB_JAR_PATH = '../ranklib/RankLib-2.1-patched.jar'


EPS = 1e-10


def ranklib_train_model(train_path,
                        ranker,
                        output,
                        metric_2_t='P@1',
                        metric_2_T='P@1',
                        g_max=1,
                        norm='sum',
                        t_v_s=1.0,
                        model_params=None,
                        ranklib_path=RANKLIB_JAR_PATH,
                        java_path=JAVA_CMD_PATH,
                        java_opts=JAVA_CMD_OPTS):

    params_list = []
    if model_params:
        for k, v in model_params.items():
            params_list.append(str(k))
            if v != '':
                params_list.append(str(v))

    proc_args = [java_path,
                 java_opts,
                 ranklib_path,
                 '-train', train_path,
                 '-ranker', str(ranker),
                 '-metric2t', metric_2_t,
                 '-metric2T', metric_2_T,
                 '-tvs', str(t_v_s),
                 '-gmax', str(g_max),
                 # '-norm', str(norm),
                 '-save', output,
                 ]
    proc_args += params_list
    ranklib_proc = subprocess.Popen(proc_args,
                                    stdout=subprocess.PIPE)

    print(proc_args)
    print('Starting training:')
    proc_start_t = perf_counter()
    stdout_val, err = ranklib_proc.communicate()
    proc_end_t = perf_counter()
    print('Training ended in ', proc_end_t - proc_start_t)

    return stdout_val, err

# > java -jar bin/RankLib.jar -load mymodel.txt -test MQ2008/Fold1/test.txt -metric2T ERR@10


def ranklib_eval(model_path,
                 test_path,
                 metric_2_T,
                 output,
                 g_max=1,
                 norm='sum',
                 ranklib_path=RANKLIB_JAR_PATH,
                 java_path=JAVA_CMD_PATH,
                 java_opts=JAVA_CMD_OPTS,
                 normalize=True):

    eval_output, eval_err = ranklib_rank_model(model_path,
                                               test_path=test_path,
                                               metric_2_T=metric_2_T,
                                               output=output,
                                               g_max=g_max,
                                               norm=norm,
                                               ranklib_path=ranklib_path,
                                               java_path=java_path,
                                               java_opts=java_opts)

    # print('OUTPUT:', eval_output)
    print('ERROR:', eval_err)

    #
    # retrieve preds
    preds = retrieve_preds(output, normalize=normalize)
    return preds


def ranklib_rank_model(model_path,
                       test_path,
                       metric_2_T,
                       output,
                       g_max=1,
                       norm='sum',
                       ranklib_path=RANKLIB_JAR_PATH,
                       java_path=JAVA_CMD_PATH,
                       java_opts=JAVA_CMD_OPTS):

    proc_args = [java_path,
                 java_opts,
                 ranklib_path,
                 '-load', model_path,
                 '-rank', test_path,
                 # '-norm', str(norm),
                 '-gmax', str(g_max),
                 '-metric2T', metric_2_T,
                 '-score', output]
    print(proc_args)
    ranklib_proc = subprocess.Popen(proc_args,
                                    stdout=subprocess.PIPE)

    stdout_val, err = ranklib_proc.communicate()

    return stdout_val, err

# java -jar RankLib-2.1-patched.jar -load mymodel.txt -rank
# '/home/valerio/Petto
# Redigi/kaggle/allen-ai-science-qa/data/20_valid_preds_liblinear_format'
# -metric2T P@1 -score scores.txt


def retrieve_preds(preds_file_path,
                   sep='\t',
                   n_answers=4,
                   normalize=True):

    preds_frame = pandas.read_csv(preds_file_path,
                                  sep=sep,
                                  header=None,
                                  names=['questionId', 'answerId', 'rank'])
    #
    # sort by question id, then answer id
    preds_frame = preds_frame.sort(['questionId', 'answerId'])

    #
    # get the rankings
    rankings = preds_frame['rank'].values

    n_pairs = rankings.shape[0]
    print('Loaded rankings {0}'.format(n_pairs))

    n_items = n_pairs // n_answers

    #
    # reshape into a matrix
    rankings = rankings.reshape(n_items, n_answers)
    print('Reshaped into {0}'.format(rankings.shape))

    if normalize:
        rankings = normalize_preds(rankings, cos_sim=False, eps=EPS)
    #
    #
    return rankings


def ranklib_train_eval(train_path,
                       train_ids,
                       test_path,
                       test_ids,
                       correct_answers,
                       ranker,
                       output,
                       metric_2_t='P@1',
                       metric_2_T='P@1',
                       g_max=1,
                       norm='sum',
                       # reg=0.001,
                       t_v_s=1.0,
                       model_params=None,
                       ranklib_path=RANKLIB_JAR_PATH,
                       java_path=JAVA_CMD_PATH,
                       java_opts=JAVA_CMD_OPTS,
                       normalize=True):

    params_list = []
    if model_params:
        for k, v in model_params.items():
            params_list.append(str(k))
            params_list.append(str(v))

    os.makedirs(output, exist_ok=True)

    prefix_str = '{0}_{1}_{2}_{3}'.format(ranker,
                                          norm,
                                          # reg,
                                          t_v_s,
                                          '_'.join([str(c) for c in params_list]))
    #
    # train model first
    model_name = os.path.join(output, prefix_str) + '.model'

    train_output, train_err = ranklib_train_model(train_path=train_path,
                                                  ranker=ranker,
                                                  output=model_name,
                                                  metric_2_t=metric_2_t,
                                                  metric_2_T=metric_2_T,
                                                  g_max=g_max,
                                                  norm=norm,
                                                  # reg=reg,
                                                  t_v_s=t_v_s,
                                                  model_params=model_params,
                                                  ranklib_path=ranklib_path,
                                                  java_path=java_path,
                                                  java_opts=java_opts)

    # print('OUTPUT:', train_output)
    print('ERROR:', train_err)

    #
    # evaluate it on training
    train_file_name = os.path.join(output, prefix_str) + '.train.preds'
    eval_output, eval_err = ranklib_rank_model(model_name,
                                               test_path=train_path,
                                               metric_2_T=metric_2_T,
                                               output=train_file_name,
                                               g_max=g_max,
                                               norm=norm,
                                               ranklib_path=ranklib_path,
                                               java_path=java_path,
                                               java_opts=java_opts)

    # print('OUTPUT:', eval_output)
    print('ERROR:', eval_err)

    #
    # retrieve preds
    train_preds = retrieve_preds(train_file_name, normalize=normalize)

    hard_train_preds = hard_preds(train_preds)

    train_pred_file = os.path.join(output, prefix_str) + '.train.scores'
    save_predictions(train_preds, train_pred_file, ids=train_ids)

    train_pred_file = os.path.join(output, prefix_str) + '.train.submission'
    create_submission(numbers_to_letters(hard_train_preds),
                      output=train_pred_file,
                      ids=train_ids)

    train_acc = compute_accuracy(correct_answers, hard_train_preds)
    print('TRAIN:', train_acc)

    #
    # evaluate on test
    valid_file_name = os.path.join(output, prefix_str) + '.valid.preds'
    valid_output, err = ranklib_rank_model(model_name,
                                           test_path,
                                           metric_2_T=metric_2_T,
                                           output=valid_file_name,
                                           g_max=g_max,
                                           norm=norm,
                                           ranklib_path=ranklib_path,
                                           java_path=java_path,
                                           java_opts=java_opts)
    valid_preds = retrieve_preds(valid_file_name, normalize=normalize)

    hard_valid_preds = hard_preds(valid_preds)

    valid_pred_file = os.path.join(output, prefix_str) + '.valid.scores'
    save_predictions(valid_preds, valid_pred_file, ids=test_ids)

    valid_pred_file = os.path.join(output, prefix_str) + '.valid.submission'
    create_submission(numbers_to_letters(hard_valid_preds),
                      output=valid_pred_file,
                      ids=test_ids)

    return model_name, (train_preds, valid_preds)


def get_cv_results(cv_output_str, n_folds=10, verbose=True):

    if verbose:
        print(cv_output_str)

    fold_accs = []
    for i in range(n_folds):
        fold_scores = re.search('Fold\s{0}\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)'.format(i + 1),
                                cv_output_str, re.IGNORECASE)
        if fold_scores:
            fold_accs.append((fold_scores.group(1), fold_scores.group(2)))

    # print(fold_accs)
    fold_accs = [(float(a), float(b)) for a, b in fold_accs]

    avg_scores = re.search(
        'Avg.\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)', cv_output_str, re.IGNORECASE)
    avg_train = None
    avg_valid = None
    if avg_scores:
        avg_train = avg_scores.group(1)
        avg_valid = avg_scores.group(2)

    # print(avg_train, avg_valid)

    total_scores = re.search('Total\s+\|\s+\|\s+(\d+\.\d+)', cv_output_str, re.IGNORECASE)
    total_valid = None
    if total_scores:
        total_valid = total_scores.group(1)

    # print(total_valid)
    assert total_valid == avg_valid

    return fold_accs, (float(avg_train), float(avg_valid))


def ranklib_cv(train_path,
               ranker,
               output,
               n_folds,
               metric_2_t='P@1',
               metric_2_T='P@1',
               g_max=1,
               norm='sum',
               t_v_s=1.0,
               model_params=None,
               ranklib_path=RANKLIB_JAR_PATH,
               java_path=JAVA_CMD_PATH,
               java_opts=JAVA_CMD_OPTS,
               normalize=True,
               verbose=True):

    #
    # composing the message, exectugin the cmd
    if model_params is None:
        model_params = {}

    model_params.update({'-kcv': n_folds})
    eval_out, err = ranklib_train_model(train_path=train_path,
                                        ranker=ranker,
                                        output=output,
                                        metric_2_t=metric_2_t,
                                        metric_2_T=metric_2_T,
                                        g_max=g_max,
                                        norm=norm,
                                        t_v_s=t_v_s,
                                        model_params=model_params,
                                        ranklib_path=ranklib_path,
                                        java_path=java_path,
                                        java_opts=java_opts)

    if err is None:
        fold_accs, (train_acc, valid_acc) = get_cv_results(eval_out.decode("utf-8"),
                                                           n_folds,
                                                           verbose)
    else:
        raise ValueError('Something got an error {}'.format(err))

    return fold_accs, (train_acc, valid_acc)


def ranklib_cv_grid_search(train_path,
                           ranker,
                           out_path,
                           n_folds,
                           params,
                           metric_2_t='P@1',
                           metric_2_T='P@1',
                           g_max=1,
                           norm='sum',
                           t_v_s=1.0,
                           ranklib_path=RANKLIB_JAR_PATH,
                           java_path=JAVA_CMD_PATH,
                           java_opts=JAVA_CMD_OPTS,
                           normalize=True,
                           verbose=True):

    os.makedirs(out_path, exist_ok=True)
    out_log_path = os.path.join(out_path, 'exp.log')

    best_valid_acc = 0.0

    best_state = {}
    best_model = None

    sorted_param_list = sorted(params)

    configurations = None
    configurations = [dict(zip(sorted_param_list, prod))
                      for prod in itertools.product(*(params[param]
                                                      for param in sorted_param_list))]

    with open(out_log_path, 'w') as out_log:

        preamble = 'id\t' + 'ranker\t' + '\ttvs' + '\t'.join([param
                                                              for param in sorted(params.keys())]) +\
            '\ttrain-acc\ttest-acc\tfold-accs'

        print(preamble)
        out_log.write(preamble + '\n')
        out_log.flush()

        grid_count = 0

        for config in configurations:

            # print('config', config)
            fold_accs, (train_acc, valid_acc) = ranklib_cv(train_path,
                                                           ranker,
                                                           output=out_path,
                                                           n_folds=n_folds,
                                                           metric_2_t=metric_2_t,
                                                           metric_2_T=metric_2_T,
                                                           g_max=g_max,
                                                           norm=norm,
                                                           t_v_s=t_v_s,
                                                           model_params=config,
                                                           ranklib_path=ranklib_path,
                                                           java_path=java_path,
                                                           java_opts=java_opts,
                                                           normalize=normalize,
                                                           verbose=verbose)

            #
            # storing best
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_state = config

            #
            # saving
            param_str = '\t'.join([str(config[p]) for p in sorted(config)])
            fold_accs_str = '\t'.join([str(v_a) for tr_a, v_a in fold_accs])
            config_str = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(grid_count,
                                                                      ranker,
                                                                      t_v_s,
                                                                      param_str,
                                                                      train_acc,
                                                                      valid_acc,
                                                                      fold_accs_str)
            out_log.write(config_str)
            out_log.flush()
            print(config_str)

            grid_count += 1

        print('Grid ended')
        print('BEST CONFIG', best_state)
        print('BEST ACC', best_valid_acc)

        out_log.write('{}'.format(best_state))
        out_log.flush()

        return best_valid_acc, best_state


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


def pair_matrix_to_ranklib_format(pair_matrix, labels, output, n_answers=4):

    n_pairs = pair_matrix.shape[0]
    n_features = pair_matrix.shape[1]

    if labels is not None:
        assert len(labels) == n_pairs

    with open(output, 'w') as output_file:

        for i in range(n_pairs):

            feature_str = ' '.join(['{0}:{1}'.format(j, pair_matrix[i, j])
                                    for j in range(n_features)])
            if labels is not None:
                label = labels[i]
            else:
                label = 0
            line = '{0} quid:{1} {2}\n'.format(label,
                                               i // n_answers + 1,
                                               feature_str)

            output_file.write(line)


def create_groups(dataset, n_answers):
    n_items = dataset.shape[0]
    return numpy.array([n_answers for i in range(n_items // n_answers)])


def from_pair_matrix_to_preds(pair_matrix, n_answers):
    n_pairs = pair_matrix.shape[0]
    return pair_matrix.reshape(n_pairs // n_answers, n_answers)


# def prediction_tensor_into_pairwise_matrix(preds):
#     n_questions = preds.shape[0]
#     n_answers = preds.shape[1]
#     n_predictors = preds.shape[2]

#     pred_matrix = numpy.zeros((n_questions * n_answers, n_predictors))

#     for i in range(n_questions):
#         for j in range(n_answers):
#             for k in range(n_predictors):
#                 pred_matrix[i * n_answers + j, k] = preds[i, j, k]

#     return pred_matrix


def vec_to_pair_vec(vec, n_answers=4):
    n_items = vec.shape[0]
    pair_vec = numpy.zeros((n_items, n_answers),  dtype=vec.dtype)
    for i in range(n_items):
        pair_vec[i, vec[i]] = 1

    return pair_vec.reshape(n_items * n_answers)


def pairs_to_vec(pair_vec, n_answers=4):
    n_pairs = pair_vec.shape[0]

    vec = pair_vec.reshape(n_pairs // n_answers, n_answers)
    return numpy.argmax(vec, axis=1)


def ranklib_cv_grid_search_f(train,
                             train_labels,
                             train_frame_ids,
                             ranker,
                             out_path,
                             n_folds,
                             params,
                             metric_2_t='P@1',
                             metric_2_T='P@1',
                             g_max=1,
                             norm='sum',
                             t_v_s=1.0,
                             ranklib_path=RANKLIB_JAR_PATH,
                             java_path=JAVA_CMD_PATH,
                             java_opts=JAVA_CMD_OPTS,
                             normalize=True,
                             verbose=True,
                             seed=1337,
                             n_answers=4,
                             dataset_str=None):

    os.makedirs(out_path, exist_ok=True)
    out_log_path = os.path.join(out_path, 'exp.log')

    best_valid_acc = 0.0

    best_state = {}
    best_model = None

    sorted_param_list = sorted(params)

    configurations = None
    configurations = [dict(zip(sorted_param_list, prod))
                      for prod in itertools.product(*(params[param]
                                                      for param in sorted_param_list))]

    kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

    fold_train_accs = [None for i in range(n_folds)]
    fold_test_accs = [None for i in range(n_folds)]

    with open(out_log_path, 'w') as out_log:

        preamble = 'id\t' + 'ranker\t' + '\ttvs' + '\t'.join([param
                                                              for param in sorted(params.keys())]) +\
            '\ttrain-acc\ttest-acc\tfold-accs'

        print(preamble)
        out_log.write(preamble + '\n')
        out_log.flush()

        grid_count = 0

        for config in configurations:

            for k, (t_ids, v_ids) in enumerate(kf):
                train_x = prediction_tensor_into_pairwise_matrix(train[t_ids])
                train_y = vec_to_pair_vec(train_labels[t_ids], n_answers)
                train_ids = train_frame_ids[t_ids]

                test_x = prediction_tensor_into_pairwise_matrix(train_preds[v_ids])
                test_y = vec_to_pair_vec(train_labels[v_ids], n_answers)
                test_ids = train_frame_ids[v_ids]

                train_labels_fold = train_labels[t_ids]
                test_labels_fold = train_labels[v_ids]

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

                rank_model_path, (train_preds_fold, valid_preds_fold) = \
                    ranklib_train_eval(temp_train_path,
                                       train_ids,
                                       temp_valid_path,
                                       test_ids,
                                       train_labels_fold,
                                       ranker,
                                       output=out_path,
                                       t_v_s=t_v_s,
                                       normalize=True,
                                       model_params=config)

                hard_train_preds_fold = hard_preds(train_preds_fold)
                train_perf = compute_accuracy(train_labels_fold, hard_train_preds_fold)

                hard_valid_preds_fold = hard_preds(valid_preds_fold)
                valid_perf = compute_accuracy(test_labels_fold, hard_valid_preds_fold)

                fold_train_accs[k] = train_perf
                fold_test_accs[k] = valid_perf
                print('>>> TRAIN:{0} TEST:{1} <<<\n'.format(train_perf,
                                                            valid_perf))

            train_acc = sum(fold_train_accs) / n_folds
            valid_acc = sum(fold_test_accs) / n_folds
            print('\n>>> AVG TRAIN: {0} AVG TEST: {1} <<<\n'.format(train_acc, valid_acc))
            # # print('config', config)
            # fold_accs, (train_acc, valid_acc) = ranklib_cv(train_path,
            #                                                ranker,
            #                                                output=out_path,
            #                                                n_folds=n_folds,
            #                                                metric_2_t=metric_2_t,
            #                                                metric_2_T=metric_2_T,
            #                                                g_max=g_max,
            #                                                norm=norm,
            #                                                t_v_s=t_v_s,
            #                                                model_params=config,
            #                                                ranklib_path=ranklib_path,
            #                                                java_path=java_path,
            #                                                java_opts=java_opts,
            #                                                normalize=normalize,
            #                                                verbose=verbose)

            #
            # storing best
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_state = config

            #
            # saving
            param_str = '\t'.join([str(config[p]) for p in sorted(config)])
            fold_accs_str = '\t'.join([str(v_a) for tr_a, v_a in zip(fold_train_accs,
                                                                     fold_test_accs)])
            config_str = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(grid_count,
                                                                      ranker,
                                                                      t_v_s,
                                                                      param_str,
                                                                      train_acc,
                                                                      valid_acc,
                                                                      fold_accs_str)
            out_log.write(config_str)
            out_log.flush()
            print(config_str)

            grid_count += 1

        print('Grid ended')
        print('BEST CONFIG', best_state)
        print('BEST ACC', best_valid_acc)

        out_log.write('{}'.format(best_state))
        out_log.flush()

        return best_valid_acc, best_state


class RanklibGroupEnsemble():

    def __init__(self, base_model, base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params

    def fit(self, group_x, group_y, group_list, verbose=False):

        assert len(group_x) == len(group_y)
        assert len(group_x) == len(group_list)

        self.models = []
        #
        # for each group instantiate a model
        for i, (x, y, g) in enumerate(zip(group_x, group_y, group_list)):

            #
            # we shall create a file to store the data

            model = self._base_model(**self._base_model_params)

            model.fit(x, y)
            self.models.append(model)

    def predict(self, group_x, aggr_func=numpy.sum):

        assert hasattr(self, 'models')

        n_groups = len(group_x)
        n_items = group_x[0].shape[0]
        n_predictors = len(self.models)
        preds = numpy.zeros((n_groups, n_items, n_predictors))
        for i, x in enumerate(group_x):

            # x_preds = []
            for j, model in enumerate(self.models):
                pred = model.predict(x)
                #
                # normalizing
                # tra_pred = (pred - pred.min() + 1e-10) / (pred.max() - pred.min())
                # pred = tra_pred / tra_pred.sum()
                preds[i, :, j] = pred

            #
            # aggregating

        # preds = normalize_preds(preds, cos_sim=False, eps=1e-10)
        aggr_preds = aggr_func(preds, axis=2)
        # preds.append(x_pred)

        return aggr_preds


if __name__ == '__main__':

    RANKER_DICT = {'mart': 0,
                   'ranknet': 1,
                   'rankboost': 2,
                   'adarank': 3,
                   'ca': 4,
                   'lambdarank': 5,
                   'lambdamart': 6,
                   'listnet': 7,
                   'randomforest': 8}

    parser = argparse.ArgumentParser()

    parser.add_argument('predictions', type=str, nargs='+',
                        help='A series of directories containing predictions')

    parser.add_argument('-k', '--n-folds', type=int, nargs='?',
                        default=10,
                        help='Number of folds for cv')

    parser.add_argument('--test', type=str, nargs='+',
                        default=None,
                        help='Test set path')

    parser.add_argument('--train-frame', type=str,
                        default='../data/training_set.tsv',
                        help='Train set frame path')

    parser.add_argument('--test-frame', type=str,
                        default='../data/validation_set.tsv',
                        help='Test set frame path')

    parser.add_argument('-r', '--ranker',  type=str,
                        default='mart',
                        help='ranking algorithm')

    parser.add_argument('--metric2t',  type=str,
                        default='P@1',
                        help='Metric for training')

    parser.add_argument('--metric2T',  type=str,
                        default='P@1',
                        help='Metric for testing')

    parser.add_argument('--norm',  type=str,
                        default='sum',
                        help='Normalizing')

    parser.add_argument('-g', '--g-max',  type=int,
                        default=1,
                        help='max err val')

    parser.add_argument('--tvs',  type=float,
                        default=0.8,
                        help='Train valid split')

    parser.add_argument('--epoch',  type=int, nargs='+',
                        default=None,
                        help='Ranknet epoch num')

    parser.add_argument('--layer',  type=int, nargs='+',
                        default=None,
                        help='Ranknet layer num')

    parser.add_argument('--node',  type=int, nargs='+',
                        default=None,
                        help='Ranknet node num')

    parser.add_argument('--lr',  type=float, nargs='+',
                        default=None,
                        help='Ranknet learning rate')

    parser.add_argument('--round',  type=int, nargs='+',
                        default=None,
                        help='Rankboost|Adarank round num')

    parser.add_argument('--tc',  type=int, nargs='+',
                        default=None,
                        help='Rankboost num threshold candidate')

    parser.add_argument('--noeq',  type=int, nargs='+',
                        default=None,
                        help='Adarank no enqueuing strong features')

    parser.add_argument('--tolerance',  type=float, nargs='+',
                        default=None,
                        help='Adarank tolerance between two consecutive rounds of learning (default=0.002)')

    parser.add_argument('--max',  type=int, nargs='+',
                        default=None,
                        help='Adarank max num times consider feature')

    parser.add_argument('--r',  type=int, nargs='+',
                        default=None,
                        help='CA The number of random restarts (default=5)')

    parser.add_argument('--i',  type=int, nargs='+',
                        default=None,
                        help='CA Num iterations (default=25)')

    parser.add_argument('--reg',  type=float, nargs='+',
                        default=None,
                        help='CA regularizaton')

    parser.add_argument('--tree',  type=int, nargs='+',
                        default=None,
                        help='(Lambda)Mart num trees')

    parser.add_argument('--leaf',  type=int, nargs='+',
                        default=None,
                        help='(Lambda)Mart max num leaves')

    parser.add_argument('--shrinkage',  type=float, nargs='+',
                        default=None,
                        help='(Lambda)Mart shrinkage factor')

    parser.add_argument('--mls',  type=int, nargs='+',
                        default=None,
                        help='(Lambda)Mart min num leaf support')

    parser.add_argument('--estop',  type=int, nargs='+',
                        default=None,
                        help='(Lambda)Mart early stopping (300)')

    parser.add_argument('--bag',  type=int, nargs='+',
                        default=None,
                        help='RF num bagged')

    parser.add_argument('--srate',  type=float, nargs='+',
                        default=None,
                        help='RF sampling rate')

    parser.add_argument('--frate',  type=float, nargs='+',
                        default=None,
                        help='RF feature sampling rate')

    parser.add_argument('--seed', type=int, nargs='+',
                        default=[1234],
                        help='Seeding for random number generation')

    parser.add_argument('--verbose', type=int,
                        default=0,
                        help='Verbosity')

    parser.add_argument('-n', '--negative-sampling', type=int, nargs='+',
                        default=[0],
                        help='Negative sampling, if 0 hierarchical softmax')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/l2r/',
                        help='Output dir path')

    parser.add_argument('--train-ext', type=str,
                        default=TRAIN_PREDS_EXT,
                        help='Train preds file extension')

    parser.add_argument('--valid-ext', type=str,
                        default=VALID_PREDS_EXT,
                        help='Valid preds file extension')

    parser.add_argument('--train-set', type=str,
                        default='../data/training_set.tsv',
                        help='Training set path')

    parser.add_argument('--aug', type=str, nargs='?',
                        default=None,
                        help='Augment features (log|ens|len)')

    args = parser.parse_args()

    print("Starting with arguments:", args)

    #
    # opening log file
    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.output + 'exp_{}_'.format(args.ranker) + date_string
    out_log_path = out_path + '/exp.log'

    #
    # creating params values dict
    params_dict = {}

    if args.epoch is not None:
        params_dict.update({'-epoch': args.epoch})

    if args.layer is not None:
        params_dict.update({'-layer': args.layer})

    if args.node is not None:
        params_dict.update({'-node': args.node})

    if args.lr is not None:
        params_dict.update({'-lr': args.lr})

    if args.round is not None:
        params_dict.update({'-round': args.round})

    if args.tc is not None:
        params_dict.update({'-tc': args.tc})

    if args.noeq is not None:
        params_dict.update({'-noeq': ['']})

    if args.tolerance is not None:
        params_dict.update({'-tolerance': args.tolerance})

    if args.max is not None:
        params_dict.update({'-max': args.max})

    if args.r is not None:
        params_dict.update({'-r': args.r})

    if args.i is not None:
        params_dict.update({'-i': args.i})

    if args.reg is not None:
        params_dict.update({'-reg': args.reg})

    if args.tree is not None:
        params_dict.update({'-tree': args.tree})

    if args.leaf is not None:
        params_dict.update({'-leaf': args.leaf})

    if args.shrinkage is not None:
        params_dict.update({'-shrinkage': args.shrinkage})

    if args.mls is not None:
        params_dict.update({'-mls': args.mls})

    if args.estop is not None:
        params_dict.update({'-estop': args.estop})

    if args.bag is not None:
        params_dict.update({'-bag': args.bag})

    if args.srate is not None:
        params_dict.update({'-srate': args.srate})

    if args.frate is not None:
        params_dict.update({'-frate': args.frate})

    verbose = True if args.verbose > 0 else False

    if args.valid_ext.lower() == 'none':
        args.valid_ext = None

    #
    # getting predictions as a tensor
    train_preds = collect_predictions_from_dirs(args.predictions, args.train_ext)
    valid_preds = None
    if args.valid_ext is not None:
        valid_preds = collect_predictions_from_dirs(args.predictions, args.valid_ext)

    if args.aug == 'log':
        train_preds = augment_predictions(train_preds, aggr_funcs=[], normalize=False, logify=True)
        if args.valid_ext is not None:
            valid_preds = augment_predictions(
                valid_preds, aggr_funcs=[], normalize=False, logify=True)
    elif args.aug == 'ens':
        train_preds = augment_predictions(train_preds, normalize=False)
        if args.valid_ext is not None:
            valid_preds = augment_predictions(valid_preds, normalize=False)
    elif args.aug == 'len':
        train_preds = augment_predictions(train_preds,  normalize=False, logify=True)
        if args.valid_ext is not None:
            valid_preds = augment_predictions(valid_preds,  normalize=False, logify=True)
    elif args.aug == 'loo':
        train_preds = numpy.log1p(train_preds)
        if args.valid_ext is not None:
            valid_preds = numpy.log1p(valid_preds)

    #
    # getting labels
    train_frame = load_data_frame(args.train_set)
    train_labels = get_answers(train_frame, numeric=True)
    train_frame_ids = numpy.array(get_ids(train_frame))

    #
    # from tensors to pair matrices
    train_pair_matrix = prediction_tensor_into_pairwise_matrix(train_preds)
    train_labels_pairs = vec_to_pair_vec(train_labels)
    if args.valid_ext is not None:
        valid_pair_matrix = prediction_tensor_into_pairwise_matrix(valid_preds)

    #
    # creating a temp file containing the data in ranklib format
        param_str = '_'.join(str(k) + '#' + '%'.join(str(v) for v in params_dict[k])
                             for k in sorted(params_dict))

    os.makedirs(TMP_PATH, exist_ok=True)

    dataset_str = '+'.join(str(p.split('/')[-2]) for p in args.predictions)
    temp_train_file = '{0}_{1}.train.pair.matrix'.format(dataset_str,
                                                         param_str)
    temp_train_path = os.path.join(TMP_PATH, temp_train_file)

    if args.valid_ext is not None:
        temp_valid_file = '{0}_{1}.valid.pair.matrix'.format(dataset_str,
                                                             param_str)
        temp_valid_path = os.path.join(TMP_PATH, temp_valid_file)

    #
    # and writing to it
    pair_matrix_to_ranklib_format(train_pair_matrix, train_labels_pairs, temp_train_path)
    if args.valid_ext is not None:
        pair_matrix_to_ranklib_format(valid_pair_matrix, None, temp_valid_path)

    # print(params_dict)
    if args.n_folds > 1:

        #
        # starting the grid

        best_acc, best_params = ranklib_cv_grid_search_f(train_preds,
                                                         train_labels,
                                                         train_frame_ids,
                                                         RANKER_DICT[args.ranker],
                                                         out_path,
                                                         args.n_folds,
                                                         params_dict,
                                                         metric_2_t=args.metric2t,
                                                         metric_2_T=args.metric2T,
                                                         g_max=args.g_max,
                                                         norm=args.norm,
                                                         t_v_s=args.tvs,
                                                         ranklib_path=RANKLIB_JAR_PATH,
                                                         java_path=JAVA_CMD_PATH,
                                                         java_opts=JAVA_CMD_OPTS,
                                                         normalize=True,
                                                         verbose=verbose)

        #
        # now training it on all training
        if '-kcv' in best_params:
            best_params.pop('-kcv')

        best_out_path = os.path.join(out_path, 'best/')

    else:
        best_params = {k: v[0] for k, v in params_dict.items()}
        best_out_path = out_path

    train_frame = load_data_frame(args.train_frame)
    valid_frame = load_data_frame(args.test_frame)

    train_ids = get_ids(train_frame)
    valid_ids = get_ids(valid_frame)

    correct_answers = get_answers(train_frame, numeric=True)

    ranklib_train_eval(temp_train_path,
                       train_ids,
                       temp_valid_path,
                       valid_ids,
                       correct_answers,
                       RANKER_DICT[args.ranker],
                       best_out_path,
                       model_params=best_params,
                       metric_2_t=args.metric2t,
                       metric_2_T=args.metric2T,
                       g_max=args.g_max,
                       norm=args.norm,
                       t_v_s=args.tvs,
                       ranklib_path=RANKLIB_JAR_PATH,
                       java_path=JAVA_CMD_PATH,
                       java_opts=JAVA_CMD_OPTS)

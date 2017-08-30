import numpy
import scipy
import scipy.stats

from dataset import load_data_frame
from dataset import get_answers
from dataset import get_ids
from dataset import save_predictions
from dataset import numbers_to_letters
from dataset import load_tensors_preds_from_files


from evaluation import compute_accuracy
from evaluation import create_submission
from evaluation import hard_preds
from evaluation import normalize_preds
from evaluation import norm_sum_preds


from ensemble import collect_predictions_from_dirs
from ensemble import predictions_names_from_dirs
from ensemble import Calibrator
from ensemble import weighted_averaging_predictions_unnorm

from sklearn import linear_model
# from ensemble import averaging_predictions
# from ensemble import ranked_averaging_predictions
# from ensemble import geom_averaging_predictions
# from ensemble import harm_averaging_predictions
# from ensemble import majority_vote
# from ensemble import weighted_ranked_averaging_predictions
# from ensemble import median_predictions
# from ensemble import circular_averaging_predictions
# from ensemble import std_averaging_predictions
from ensemble import aggregate_function_factory

# from ensemble import hard_averaging_predictions
# from ensemble import hard_ranked_averaging_predictions
# from ensemble import hard_geom_averaging_predictions
# from ensemble import hard_harm_averaging_predictions
# from ensemble import majority_vote
# from ensemble import hard_weighted_ranked_averaging_predictions
# from ensemble import hard_median_predictions

from time import perf_counter

import argparse

import logging

import itertools

import os

import datetime

import seaborn
import matplotlib.pyplot as pyplot

# def aggregate_predictions_by_averaging(preds):
#     print('Averaging predictions')
#     avg_preds = averaging_predictions(preds)
#     avg_preds = hard_preds(avg_preds)
#     return avg_preds


# def aggregate_predictions_by_majority_vote(preds):
#     print('Majority vote')
#     return majority_vote(preds)

# AGGR_FUNC_DICT = {
#     'avg': hard_averaging_predictions,
#     'maj': majority_vote,
#     'rnk': hard_ranked_averaging_predictions,
#     'geo': hard_geom_averaging_predictions,
#     'hrm': hard_harm_averaging_predictions,
#     'wrk': hard_weighted_ranked_averaging_predictions,
#     'med': hard_median_predictions,
# }


TRAIN_SUBS_EXT = 'train.submission'
VALID_SUBS_EXT = 'valid.submission'

TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'

OUT_PREDS_EXT = '.preds'


def compute_aggregation(preds, aggr_func_type):
    # print('Dealing with pred matrix of {}'.format(preds.shape))

    aggr_func = aggregate_function_factory(aggr_func_type)
    # aggr_func = AGGR_FUNC_DICT[aggr_func_type]

    aggr_preds = aggr_func(preds)

    assert len(aggr_preds) == len(preds)
    # print('Reduced to', aggr_preds.shape)

    return aggr_preds


def compute_weighted_avg(preds, weights):
    assert (preds.shape[2] == len(weights))

    return weighted_averaging_predictions_unnorm(preds, weights)


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


def compute_accuracies(hard_preds, correct_answers):

    assert hard_preds.shape[0] == correct_answers.shape[0]

    assert hard_preds.ndim == 2

    n_predictors = hard_preds.shape[1]

    accuracies = [compute_accuracy(correct_answers, hard_preds[:, i])
                  for i in range(n_predictors)]

    return numpy.array(accuracies)


def train_answer_coverage(hard_preds, train_answers):

    assert hard_preds.ndim == 2

    n_questions = len(train_answers)
    n_predictors = hard_preds.shape[1]

    stats = numpy.zeros(n_questions, dtype=int)

    for i in range(n_questions):
        for j in range(n_predictors):
            if hard_preds[i, j] == train_answers[i]:
                stats[i] += 1

    #
    # print how many questions are not covered by at least X models
    print('# preds\t# questions not covered\t(%)')
    for j in range(0, n_predictors + 1):
        uncovered_questions = stats < j + 1
        covered_questions = stats > j
        print('{0}\t{1} \t({2}) \t{3} \t({4})'.format(j,
                                                      sum(uncovered_questions),
                                                      sum(uncovered_questions) / n_questions,
                                                      sum(covered_questions),
                                                      sum(covered_questions) / n_questions))
    return stats


def visualize_interactions(int_matrix, names, mask):

    f, ax = pyplot.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    h = seaborn.heatmap(int_matrix,
                        # mask=mask,
                        cmap=cmap,  # vmax=.3,
                        square=True,  # xticklabels=5, yticklabels=5,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    h.set_xticklabels(names, rotation=90)
    h.set_yticklabels(list(reversed(names)))

    pyplot.show()


def predictions_correlations(hard_preds, preds_names, mask=None):
    import pandas

    preds_frame = pandas.DataFrame(data=hard_preds,
                                   columns=preds_names)

    corr = preds_frame.corr()
    print(corr)
    # corr, p_value = scipy.stats.spearmanr(hard_preds)

    if mask:
        mask = numpy.zeros_like(corr, dtype=bool)
        mask[numpy.triu_indices_from(mask)] = True

    visualize_interactions(corr, preds_names, mask)

    return corr


def predictions_equality(hard_preds, preds_names):

    n_questions = hard_preds.shape[0]
    n_predictors = hard_preds.shape[1]

    eq_matrix = numpy.zeros((n_predictors, n_predictors))

    for i in range(n_predictors):
        eq_matrix[i, i] = 1.0

    for i, j in itertools.combinations(range(n_predictors), 2):
        eq_preds = hard_preds[:, i] == hard_preds[:, j]
        eq_matrix[i, j] = sum(eq_preds) / n_questions
        eq_matrix[j, i] = eq_matrix[i, j]

    visualize_interactions(eq_matrix, preds_names, None)

    return eq_matrix

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('predictions', type=str, nargs='+',
                        help='A series of directories containing predictions')

    parser.add_argument('--train-ext', type=str,
                        default=TRAIN_PREDS_EXT,
                        help='Train preds file extension')

    parser.add_argument('--train', type=str,
                        default='../data/training_set.tsv',
                        help='Training set path')

    parser.add_argument('--valid-ext', type=str,
                        default=VALID_PREDS_EXT,
                        help='Valid preds file extension')

    parser.add_argument('-a', '--aggr-func',  type=str, nargs='+',
                        default=['avg'],
                        help='Aggregating functions (avg, maj)')

    parser.add_argument('--id',  type=str,
                        # default='0',
                        help='Counter to identify the output')

    parser.add_argument('--cv',  type=int,
                        default=None,
                        help='Ensemble CV folds results')

    parser.add_argument('-c', '--coverage', action='store_true',
                        help='Compute train coverage')

    parser.add_argument('-p', '--pearson', action='store_true',
                        help='Compute prediction correlations')

    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Normalize prediction')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../submissions/',
                        help='Output dir path for predictions file')

    parser.add_argument('-w', '--weights',  type=float, nargs='+',
                        help='Weights for a weighted average')

    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate prediction')

    parser.add_argument('--load-matrices', action='store_true',
                        help='Load numpy matrices instead of predictions')

    args = parser.parse_args()

    print("Starting with arguments:", args)

    #
    #
    if args.valid_ext.lower() == 'none':
        args.valid_ext = None

    print('\nLoading training set')
    train_frame = load_data_frame(args.train)
    train_labels = get_answers(train_frame, numeric=True)
    train_ids = get_ids(train_frame)

    seed = 1337
    if args.cv is None:

        #
        # collecting preds in a tensor
        print('\nCollecting train predictions')
        if args.load_matrices:
            print('By appending numpy files')
            train_preds = load_tensors_preds_from_files(args.predictions, args.train_ext)
        else:
            print('By scanning directories')
            train_preds = collect_predictions_from_dirs(args.predictions, args.train_ext)

        if args.normalize:
            train_preds = norm_sum_preds(train_preds)

        hard_train_preds = hard_preds(train_preds)

        if args.valid_ext:
            print('\nCollecting valid predictions')
            if args.load_matrices:
                print('By appending numpy files')
                valid_preds = load_tensors_preds_from_files(args.predictions, args.valid_ext)
            else:
                print('By scanning directories')
                valid_preds = collect_predictions_from_dirs(args.predictions, args.valid_ext)

            if args.normalize:
                valid_preds = norm_sum_preds(valid_preds)

            hard_valid_preds = hard_preds(valid_preds)

        # train_preds = numpy.log1p(train_preds)
        # print(train_preds[:2])
        # valid_preds = numpy.log1p(valid_preds)

        caliber = None
        if args.calibrate:
            caliber = Calibrator(linear_model.LogisticRegression,
                                 fit_intercept=True,
                                 class_weight='balanced',
                                 C=0.8)
            caliber.fit(train_preds, train_labels)
            train_preds = caliber.predict(train_preds)
            valid_preds = caliber.predict(valid_preds)

        train_perfs = []

        for aggr_func in args.aggr_func:

            train_aggr_preds = None
            print('\nAggregating train with', aggr_func)
            if aggr_func == 'wvg':
                if args.weights:
                    train_aggr_preds = compute_weighted_avg(train_preds, args.weights)
                else:
                    raise ValueError('No weights specified')

            else:
                train_aggr_preds = compute_aggregation(train_preds, aggr_func)

            norm_train_aggr_preds = None
            if aggr_func == 'maj':
                hard_train_aggr_preds = train_aggr_preds
            else:
                hard_train_aggr_preds = hard_preds(train_aggr_preds)
                norm_train_aggr_preds = normalize_preds(train_aggr_preds, cos_sim=False, eps=1e-10)
                make_scores(
                    norm_train_aggr_preds, aggr_func, args.output, args.id, 'train', train_ids)

            train_acc = compute_accuracy(train_labels, hard_train_aggr_preds)

            make_submission(
                hard_train_aggr_preds, aggr_func, args.output, args.id, 'train', train_ids)

            train_perfs.append((aggr_func, train_acc))

            print('Accuracy on training', train_acc)

            if args.valid_ext:
                print('\nAggregating valid with', aggr_func)
                valid_aggr_preds = None
                if aggr_func == 'wvg':
                    if args.weights:
                        valid_aggr_preds = compute_weighted_avg(valid_preds, args.weights)
                    else:
                        raise ValueError('No weights specified')

                else:
                    valid_aggr_preds = compute_aggregation(valid_preds, aggr_func)

                norm_valid_aggr_preds = None
                if aggr_func == 'maj':
                    hard_valid_aggr_preds = valid_aggr_preds
                else:
                    hard_valid_aggr_preds = hard_preds(valid_aggr_preds)
                    norm_valid_aggr_preds = normalize_preds(
                        valid_aggr_preds, cos_sim=False, eps=1e-10)
                    make_scores(norm_valid_aggr_preds, aggr_func, args.output, args.id, 'valid')

                make_submission(hard_valid_aggr_preds, aggr_func, args.output, args.id, 'valid')

        #
        # printing all train scores
        print('\n\n**** Train accuracy ranking summary ****\n')
        for aggr_func, score in reversed(sorted(train_perfs, key=lambda tup: tup[1])):
            print('{0}\t{1}'.format(aggr_func, score))

        #
        # train coverage
        if args.coverage:
            print('\n\n**** Train coverage ****\n')
            train_answer_coverage(hard_train_preds, train_labels)

        if args.pearson:
            train_preds_names = predictions_names_from_dirs(args.predictions, args.train_ext)
            train_preds_names = [n.replace(args.train_ext, '') for n in train_preds_names]
            # print(train_preds_names)
            predictions_correlations(hard_train_preds, train_preds_names)
            predictions_correlations(hard_valid_preds, train_preds_names)
            # predictions_equality(hard_train_preds, train_preds_names)

    #
    # aggregating over folds
    else:
        raise ValueError('Be careful')

        from sklearn.cross_validation import KFold

        kf = KFold(len(train_ids), n_folds=args.cv, shuffle=True, random_state=seed)

        train_indexes = [t_id for t_id, v_id in kf]
        valid_indexes = [v_id for t_id, v_id in kf]

        fold_accs = []
        for i in range(args.cv):
            fold_preds_paths = [os.path.join(pred_dir, str(i)) for pred_dir in args.predictions]
            #
            # create fold path
            print('Examining fold {0} ({1})'.format(i, fold_preds_paths))

            train_preds = collect_predictions_from_dirs(fold_preds_paths, args.train_ext)
            valid_preds = collect_predictions_from_dirs(fold_preds_paths, args.valid_ext)

            train_perfs = []

            for aggr_func in args.aggr_func:

                print('\nAggregating train with', aggr_func)
                train_aggr_preds = compute_aggregation(train_preds, aggr_func)

                norm_train_aggr_preds = None
                if aggr_func == 'maj':
                    hard_train_aggr_preds = train_aggr_preds
                else:
                    hard_train_aggr_preds = hard_preds(train_aggr_preds)
                    norm_train_aggr_preds = normalize_preds(
                        train_aggr_preds, cos_sim=False, eps=1e-10)
                    # make_scores(norm_train_aggr_preds,
                    #             aggr_func,
                    #             args.output,
                    #             args.id,
                    #             'train',
                    #             train_ids)

                print(len(train_indexes[i]), len(hard_train_aggr_preds))
                train_acc = compute_accuracy(train_labels[train_indexes[i]], hard_train_aggr_preds)
                fold_accs.append(fold_accs)

                # make_submission(hard_train_aggr_preds,
                #    aggr_func, args.output, args.id, 'train', train_ids)

                train_perfs.append((aggr_func, train_acc))

                print('Accuracy on training', train_acc)

                if args.valid_ext:
                    print('\nAggregating valid with', aggr_func)
                    valid_aggr_preds = compute_aggregation(valid_preds, aggr_func)

                    norm_valid_aggr_preds = None
                    if aggr_func == 'maj':
                        hard_valid_aggr_preds = valid_aggr_preds
                    else:
                        hard_valid_aggr_preds = hard_preds(valid_aggr_preds)
                        norm_valid_aggr_preds = normalize_preds(
                            valid_aggr_preds, cos_sim=False, eps=1e-10)
                    #    make_scores(norm_valid_aggr_preds,
                    #                aggr_func, args.output, args.id, 'valid')

                    # make_submission(hard_valid_aggr_preds,
                    #                 aggr_func, args.output, args.id, 'valid')

            #
            # printing all train scores
            print('\n\n**** Train accuracy ranking summary ****\n')
            for aggr_func, score in reversed(sorted(train_perfs, key=lambda tup: tup[1])):
                print('{0}\t{1}'.format(aggr_func, score))

        print('Over folds, cv gen error:')
        fold_accs = numpy.array(fold_accs)
        print(fold_accs)
        print(fold_accs.mean())

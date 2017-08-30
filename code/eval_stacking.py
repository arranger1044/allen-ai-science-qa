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

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str,
                        help='Path to the stacked tensor data')

    # parser.add_argument('-', '--train', type=str,
    #                     # default='../data/training_set.tsv',
    #                     help='Training set path')

    parser.add_argument('-f', '--data-set', type=str,
                        default=None,
                        help='Data set path')

    parser.add_argument('-m', '--model', type=str,
                        help='Model pickle path')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/onevs/',
                        help='Output dir path')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    parser.add_argument('--acc', action='store_true',
                        help='Compute accuracy')

    #
    # parsing the args
    args = parser.parse_args()
    print(args)

    #
    # loading the stacked data
    preds_tensor = numpy.load(args.data)
    print('Loaded tensor with shape {}'.format(preds_tensor.shape))

    model = None
    with open(args.model, 'rb') as model_file:
        model = pickle.load(model_file)
    print('Loaded meta model {}'.format(model))

    print('Predicting...')
    model_preds = model.predict(preds_tensor)

    #
    # if a data set has been specified we can get the ids
    if args.data_set:
        data_frame = load_data_frame(args.data_set)

        frame_ids = numpy.array(get_ids(data_frame))

        date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = None
        if args.exp_name:
            out_path = os.path.join(args.output, 'exp_' + args.exp_name)
        else:
            out_path = os.path.join(args.output, 'exp_' + date_string)
        out_preds_path = os.path.join(out_path, 'data.preds')
        os.makedirs(out_path, exist_ok=True)

        save_predictions(model_preds, out_preds_path, frame_ids)

        out_preds_path = os.path.join(out_path, 'data.submission')
        hard_model_preds = hard_preds(model_preds)
        create_submission(numbers_to_letters(hard_model_preds),
                          output=out_preds_path,
                          ids=frame_ids)
        if args.acc:
            print('Computing accuracy')
            data_labels = numpy.array(get_answers(data_frame, numeric=True))
            model_acc = compute_accuracy(data_labels, hard_model_preds)
            print(model_acc)

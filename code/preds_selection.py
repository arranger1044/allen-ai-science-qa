from collections import defaultdict

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
                   'ovo': OneVsOneClassifier}

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import load_data_frame
from dataset import get_answers
from dataset import get_questions
from dataset import numbers_to_letters
from dataset import get_ids
from dataset import save_predictions

from ensemble import collect_predictions_from_dirs

import numpy
import pandas

import random

from keras.preprocessing import sequence, text
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding

from keras.layers.convolutional import Convolution1D, MaxPooling1D

import sys
import logging

import argparse


def build_query_graph(graph,
                      embedding_size=500,
                      max_features=100000,
                      max_len=100,
                      input_dropout_rate=0.25,
                      dropout_rate=0.5,
                      convolutional_kernels=16,
                      filter_extension=3):
    '''
    Builds Keras Graph model that, given a query (in the form of a list of indices),
    returns a vector of output_dim
    non-negative weights that sum up to 1.

    The Convolutional Neural Network architecture is inspired by the following paper:
    Yoon Kim - Convolutional Neural Networks for Sentence Classification - arXiv:1408.5882v2
    '''

    graph.add_input(name='input_query', input_shape=(None,), dtype='int32')

    embedding_layer = Embedding(input_dim=max_features,
                                output_dim=embedding_size,
                                input_length=max_len)

    graph.add_node(embedding_layer, name='embedding', input='input_query')

    graph.add_node(Dropout(input_dropout_rate), name='embedding_dropout', input='embedding')

    convolutional_layer = Convolution1D(input_dim=embedding_size,
                                        nb_filter=convolutional_kernels,
                                        filter_length=filter_extension,
                                        border_mode='valid',
                                        activation='relu',
                                        subsample_length=1)

    graph.add_node(convolutional_layer, name='convolutional', input='embedding_dropout')

    pool_length = convolutional_layer.output_shape[1]

    pooling_layer = MaxPooling1D(pool_length=pool_length)
    graph.add_node(pooling_layer, name='pooling', input='convolutional')

    flatten_layer = Flatten()
    graph.add_node(flatten_layer, name='flatten', input='pooling')

    graph.add_node(Dropout(dropout_rate), name='dropout', input='flatten')

    dense_layer = Dense(output_dim=output_dim)
    graph.add_node(dense_layer, name='dense', input='dropout')

    softmax_layer = Activation('softmax')
    graph.add_node(softmax_layer, name='softmax', input='dense')
    return graph


def build_tensor_graph(graph, input_shape=(4, 20)):
    graph.add_input(name='input_tensor', input_shape=input_shape)
    return graph


def train_single_convolutional_expert_mixture(train_X,
                                              train_tensor_preds,
                                              train_labels,
                                              valid_X,
                                              valid_tensor_preds,
                                              valid_labels,
                                              test_X,
                                              test_tensor_preds,
                                              optimizer='adadelta',
                                              loss='mse',
                                              embedding_size=500,
                                              max_features=100000,
                                              max_len=100,
                                              input_dropout_rate=0.25,
                                              dropout_rate=0.5,
                                              convolutional_kernels=16,
                                              filter_extension=3,
                                              batch_size=100,
                                              n_epochs=100):

    #
    # init graph
    graph = Graph()

    #
    # add query part
    graph = build_query_graph(graph)

    #
    # add tensor part
    graph = build_tensor_graph(graph)

    #
    # merging
    graph.add_node(Activation('linear'), name='dot', inputs=[
                   'softmax', 'input_tensor'], merge_mode='dot')
    graph.add_output(name='output', input='dot')

    #
    # compiling
    graph.compile(optimizer, {'output': loss})

    #
    # fitting on train
    graph.fit({'input_query': train_X,
               'input_tensor': train_tensor_preds,
               'output': train_labels},
              batch_size=batch_size,
              nb_epoch=n_epochs,
              verbose=1)

    #
    # predicting on valid
    out = graph.predict({'input_query': train_X[0, :].reshape((1, train_X.shape[1])),
                         'input_tensor': train_tensor_preds[0, :, :].reshape((1, train_tensor_preds.shape[1],
                                                                              train_tensor_preds.shape[2])),
                         }, batch_size=1)

    #
    # predicting on test

    print(out)

TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'

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

    parser.add_argument('--max-features', type=int,
                        default=100000,
                        help='Max num of words in dictionary')

    parser.add_argument('-l', '--max-length', type=int,
                        default=100,
                        help='Max num of words in a sentence')

    parser.add_argument('-e', '--embedding', type=int, nargs='+',
                        default=[500],
                        help='Embedding size')

    parser.add_argument('--optimizer', type=str, nargs='+',
                        default=['adadelta'],
                        help='Optimizer fo sgd')

    parser.add_argument('--loss', type=str, nargs='+',
                        default=['mse'],
                        help='Loss function name')

    parser.add_argument('--init-drop-rate', type=float, nargs='+',
                        default=[0.25],
                        help='Input layer dropout rate')

    parser.add_argument('--dense-drop-rate', type=float, nargs='+',
                        default=[0.5],
                        help='Dense layer dropout rate')

    parser.add_argument('-k', '--conv-kernels', type=int, nargs='+',
                        default=[16],
                        help='Num convolutional kernels')

    parser.add_argument('-x', '--filter-extension', type=int, nargs='+',
                        default=[3],
                        help='Filter length')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/conv-exp-mix/',
                        help='Output dir path')

    parser.add_argument('--cv', action='store_true',
                        help='Cross validating model')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=0,
                        help='Verbosity level')

    parser.add_argument('--train-ext', type=str,
                        default=TRAIN_PREDS_EXT,
                        help='Train preds file extension')

    parser.add_argument('--valid-ext', type=str,
                        default=VALID_PREDS_EXT,
                        help='Valid preds file extension')

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
    train_preds = collect_predictions_from_dirs(args.preds, args.train_ext)
    valid_preds = collect_predictions_from_dirs(args.preds, args.valid_ext)

    n_predictors = train_preds.shape[2]
    assert valid_preds.shape[2] == n_predictors

    output_dim = n_predictors

    logging.basicConfig(level=logging.DEBUG)

    #
    # building train and validation sequences
    train_questions = get_questions(train_frame)
    valid_questions = get_questions(valid_frame)

    # Train the tokenizer on both training and validation sets
    questions = train_questions + valid_questions

    tokenizer = text.Tokenizer(nb_words=args.max_features)
    tokenizer.fit_on_texts(questions)

    train_sequences = [seq for seq in tokenizer.texts_to_sequences_generator(train_questions)]
    valid_sequences = [seq for seq in tokenizer.texts_to_sequences_generator(valid_questions)]

    # X contains all training questions
    train_X = sequence.pad_sequences(train_sequences, maxlen=args.max_length)
    logging.info('train X is: %s' % str(train_X.shape))
    valid_X = sequence.pad_sequences(valid_sequences, maxlen=args.max_length)
    logging.info('valid X is: %s' % str(valid_X.shape))

    #
    # TODO: extend it to cv
    train_single_convolutional_expert_mixture(train_X,
                                              train_preds,
                                              train_labels,
                                              None, None, None,
                                              valid_X,
                                              valid_preds,
                                              optimizer=args.optimizer[0],
                                              loss=args.loss[0],
                                              embedding_size=args.embedding[0],
                                              max_features=args.max_features,
                                              max_len=args.max_length,
                                              input_dropout_rate=args.init_drop_rate[0],
                                              dropout_rate=args.dense_drop_rate[0],
                                              convolutional_kernels=args.conv_kernels[0],
                                              filter_extension=args.filter_extension[0])

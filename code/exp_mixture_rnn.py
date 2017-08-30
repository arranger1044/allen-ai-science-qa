#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np

import mixture.utils as utils

from dataset import load_data_frame
from dataset import get_answers
from dataset import get_questions

from ensemble import collect_predictions_from_dirs

from preprocess import preprocess_factory

from keras.preprocessing import sequence, text
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D

import inspect

import sys
import ast
import logging
import argparse
import itertools
import collections


def build_query_graph(graph, embedding_size=500, embedding_path=None, token2idx=None,
                      input_dropout_rate=0.25, dropout_rate=0.5, l1=None, l2=None,
                      model_name='rnn', hidden_size=100, fix_embeddings=False,
                      max_features=100000, max_len=100, n_predictors=80):
    '''
    Builds Keras Graph model that, given a query (in the form of a list of indices), returns a vector of output_dim
    non-negative weights that sum up to 1.

    The Recurrent Neural Network architecture is inspired by the following paper:
    Tang, Duyu et al. - Document Modeling with Gated Recurrent Neural Network for Sentiment Classification - EMNLP 2015
    '''
    regularizer = utils.get_regularizer(l1, l2)

    graph.add_input(name='input_query', input_shape=(None,), dtype='int32')

    E = None
    if embedding_path is not None:
        E = utils.read_embeddings(embedding_path, token2idx=token2idx, max_features=max_features)

    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size, input_length=max_len, weights=E)

    if fix_embeddings is True:
        embedding_layer.params = []
        embedding_layer.updates = []

    graph.add_node(embedding_layer, name='embedding', input='input_query')

    graph.add_node(Dropout(input_dropout_rate), name='embedding_dropout', input='embedding')

    recurrent_layer = utils.get_recurrent_layer(model_name, embedding_size, hidden_size)

    graph.add_node(recurrent_layer, name='recurrent', input='embedding_dropout')

    graph.add_node(Dropout(dropout_rate), name='dropout', input='recurrent')

    # Length of the output vector (provided by a softmax layer)
    output_dim = n_predictors

    dense_layer = Dense(output_dim=output_dim, W_regularizer=regularizer)
    graph.add_node(dense_layer, name='dense', input='dropout')

    softmax_layer = Activation('softmax')
    graph.add_node(softmax_layer, name='softmax', input='dense')
    return graph


def build_tensor_graph(graph, n_predictors):
    graph.add_input(name='input_tensor', input_shape=(4, n_predictors))
    return graph


def train_model(graph,
                train_X, train_tensor, train_answers, train_y,
                valid_X, valid_tensor, valid_answers, valid_y,
                optimizer='adadelta', loss='categorical_crossentropy', epochs=50, batch_size=32):
    graph.compile(optimizer=optimizer, loss={'output': loss})

    # Here startes the learning part
    best_eval_on_validation = None
    prev_best_score_on_validation, best_score_on_validation = None, None

    for epoch in range(epochs):
        logging.info('Epoch %d on %d ..' % (epoch, epochs))

        graph.fit({'input_query': train_X, 'input_tensor': train_tensor, 'output': train_y},
                  batch_size=batch_size, nb_epoch=1, verbose=0)

        train_answers_pred = graph.predict({'input_query': train_X, 'input_tensor': train_tensor}, batch_size=1)
        train_accuracy = utils.compute_accuracy(train_answers, train_answers_pred['output'])

        logging.info('Training accuracy: %s' % train_accuracy)

        valid_answers_pred = graph.predict({'input_query': valid_X, 'input_tensor': valid_tensor}, batch_size=1)
        valid_accuracy = utils.compute_accuracy(valid_answers, valid_answers_pred['output'])

        logging.info('Validation accuracy: %s' % valid_accuracy)

        is_improvement_on_validation = False

        if best_eval_on_validation is None:
            is_improvement_on_validation = True
        else:
            current_score = valid_accuracy
            if current_score > best_score_on_validation:
                is_improvement_on_validation = True

        if is_improvement_on_validation is True:
            best_eval_on_validation = valid_answers_pred
            best_epoch_on_validation = epoch

            prev_best_score_on_validation = best_score_on_validation

            best_score_on_validation = valid_accuracy

            # We just saw an improvement on the Validation set, let's also evaluate in on the test set.
            logging.debug('Epoch %d - Detected an improvement on the validation set: %s -> %s'
                          % (best_epoch_on_validation, prev_best_score_on_validation, best_score_on_validation))

    logging.info('Best accuracy on validation set: %s' % best_score_on_validation)

    return best_score_on_validation


def experiment(train_questions, train_tensor, train_answers,
               valid_questions, valid_tensor, valid_answers, seed=1,
               max_features=100000, max_len=100, embedding_size=500, embedding_path=None, fix_embeddings=False,
               input_dropout_rate=0.25, dropout_rate=0.5, l1=None, l2=None,
               model_name='rnn', hidden_size=100,
               optimizer='adadelta', loss='mse', n_predictors=80,
               preprocess_func='tl', epochs=50, batch_size=32, logger=None):

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    args_list = [(arg, values[arg]) for arg in args if arg not in ['train_questions', 'train_tensor', 'train_answers',
                                                                   'valid_questions', 'valid_tensor', 'valid_answers',
                                                                   'logger']]

    logging.info('Hyperparameters: %s' % (str(args_list)))

    random.seed(seed)
    np.random.seed(seed)

    f = preprocess_factory(preprocess_func)

    def preprocess(question):
        return ' '.join(f(question))

    train_questions = np.array([preprocess(q) for q in train_questions])
    valid_questions = np.array([preprocess(q) for q in valid_questions])

    questions = train_questions.tolist() + valid_questions.tolist()

    # Train the tokenizer on both training and validation sets
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(questions)

    token2idx = tokenizer.word_index

    train_sequences = [seq for seq in tokenizer.texts_to_sequences_generator(train_questions)]
    valid_sequences = [seq for seq in tokenizer.texts_to_sequences_generator(valid_questions)]

    # X contains all training questions
    train_X = sequence.pad_sequences(train_sequences, maxlen=max_len)
    valid_X = sequence.pad_sequences(valid_sequences, maxlen=max_len)
    logging.info('X is: %s' % str(train_X.shape))

    train_y = np.zeros(shape=(train_answers.shape[0], 4))
    train_y[np.arange(train_answers.shape[0]), train_answers] = 1

    valid_y = np.zeros(shape=(valid_answers.shape[0], 4))
    valid_y[np.arange(valid_answers.shape[0]), valid_answers] = 1

    graph = Graph()

    graph = build_query_graph(graph, embedding_size=embedding_size, embedding_path=embedding_path, token2idx=token2idx,
                              input_dropout_rate=input_dropout_rate, dropout_rate=dropout_rate, l1=l1, l2=l2,
                              model_name=model_name, hidden_size=hidden_size,
                              max_features=max_features, max_len=max_len, n_predictors=n_predictors,
                              fix_embeddings=fix_embeddings)

    graph = build_tensor_graph(graph, n_predictors=n_predictors)

    graph.add_node(Activation('linear'), name='dot', inputs=['softmax', 'input_tensor'], merge_mode='dot')
    graph.add_output(name='output', input='dot')

    valid_score = train_model(graph,
                              train_X, train_tensor, train_answers, train_y,
                              valid_X, valid_tensor, valid_answers, valid_y,
                              optimizer=optimizer, loss=loss, epochs=epochs, batch_size=batch_size)

    logging.info('%s : %s' % (valid_score, str(args_list)))

    if logger is not None:
        entry_list = [('score', valid_score)] + args_list
        entry_dict = collections.OrderedDict(entry_list)
        logger.write(entry_dict)

    return valid_score


TRAIN_PREDS_EXT = 'train.scores'
VALID_PREDS_EXT = 'valid.scores'


def main(argv):
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('preds', type=str, nargs='+', help='One or more dirs to retrieve the paths')
    parser.add_argument('-t', '--train', type=str, default='../data/training_set.tsv', help='Training set path')
    parser.add_argument('-z', '--n-folds', type=int, nargs='?', default=5, help='Number of folds for cv')
    parser.add_argument('-f', '--test', type=str, default='../data/validation_set.tsv', help='Test set path')
    parser.add_argument('--max-features', type=int, default=100000, help='Max num of words in dictionary')
    parser.add_argument('-l', '--max-length', type=int, default=100, help='Max num of words in a sentence')

    parser.add_argument('--embedding-size', type=int, nargs='+', default=[500], help='Embedding size')
    parser.add_argument('-e', '--embedding', type=str, nargs='+', default=[None], help='Path of embedding vectors')
    parser.add_argument('--fix-embeddings', type=ast.literal_eval, nargs='+', default=[False], help='Fix Embeddings')

    parser.add_argument('--epochs', type=int, nargs='+', default=[50], help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, nargs='+', default=[32], help='Batch size')

    parser.add_argument('--optimizer', type=str, nargs='+', default=['adadelta'], help='Optimizer fo sgd')
    parser.add_argument('--loss', type=str, nargs='+', default=['mse'], help='Loss function name')

    parser.add_argument('--init-drop-rate', type=float, nargs='+', default=[0.25], help='Input layer dropout rate')
    parser.add_argument('--dense-drop-rate', type=float, nargs='+', default=[0.5], help='Dense layer dropout rate')
    parser.add_argument('--l1', type=float, nargs='+', default=[0.0], help='L1 regularization weight')
    parser.add_argument('--l2', type=float, nargs='+', default=[0.0], help='L2 regularization weight')

    parser.add_argument('--proc-func',  type=str, nargs='+', default=['tl'], help='Doc preprocessing func')

    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['rnn'], help='Recurrent model name')
    parser.add_argument('-H', '--hidden-size', type=int, nargs='+', default=[100], help='Recurrent hidden size')

    parser.add_argument('-o', '--output', type=str, nargs='?', default='../models/conv-exp-mix/', help='Output dir path')
    parser.add_argument('--cv', action='store_true', help='Cross validating model')
    parser.add_argument('--seed', type=int, nargs='?', default=1337, help='Seed for the random generator')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=0, help='Verbosity level')
    parser.add_argument('--train-ext', type=str, default=TRAIN_PREDS_EXT, help='Train preds file extension')
    parser.add_argument('--valid-ext', type=str, default=VALID_PREDS_EXT, help='Valid preds file extension')

    parser.add_argument('--save', type=str, default=None, help='Path of TSV file containing results')

    # parsing the args
    args = parser.parse_args()
    print(args)

    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)
    # I shall print here all the stats

    seed = args.seed
    np.random.seed(seed)

    # Parse the command line arguments
    optimizer_list = args.optimizer
    loss_list = args.loss
    embedding_size_list = args.embedding_size
    embedding_path_list = args.embedding
    fix_embeddings_list = args.fix_embeddings
    input_dropout_rate_list = args.init_drop_rate
    dropout_rate_list = args.dense_drop_rate

    model_name_list = args.model_name
    hidden_size_list = args.hidden_size

    l1_list = args.l1
    l2_list = args.l2

    preprocess_func_list = args.proc_func
    epochs_list = args.epochs
    batch_size_list = args.batch_size

    max_features = args.max_features
    max_len = args.max_length

    save_path = args.save

    # loading frames, labels, ids
    train_frame = load_data_frame(args.train)
    valid_test_frame = load_data_frame(args.test)

    # loading preds tensors
    train_tensor = collect_predictions_from_dirs(args.preds, args.train_ext)
    valid_test_tensor = collect_predictions_from_dirs(args.preds, args.valid_ext)

    n_predictors = train_tensor.shape[2]
    assert valid_test_tensor.shape[2] == n_predictors

    # building train and validation sequences
    train_questions = np.asarray(get_questions(train_frame))
    valid_test_questions = np.asarray(get_questions(valid_test_frame))

    train_answers = get_answers(train_frame, numeric=True)

    # Group together training and validation (test) questions for training the tokenizer
    questions = train_questions.tolist() + valid_test_questions.tolist()

    # Shuffle the training set
    train_index = np.arange(len(train_questions))
    np.random.shuffle(train_index)

    train_tensor = train_tensor[train_index]
    train_questions = train_questions[train_index]
    train_answers = train_answers[train_index]

    valid_size = int(len(train_questions) / 10.)

    valid_tensor = train_tensor[-valid_size:]
    valid_questions = train_questions[-valid_size:]
    valid_answers = train_answers[-valid_size:]

    train_tensor = train_tensor[:-valid_size]
    train_questions = train_questions[:-valid_size]
    train_answers = train_answers[:-valid_size]

    def cartesian_product(dicts):
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    configuration_space = dict(optimizer=optimizer_list,
                               loss=loss_list,
                               embedding_size=embedding_size_list,
                               embedding_path=embedding_path_list,
                               fix_embeddings=fix_embeddings_list,
                               input_dropout_rate=input_dropout_rate_list,
                               dropout_rate=dropout_rate_list,

                               model_name=model_name_list,
                               hidden_size=hidden_size_list,

                               l1=l1_list, l2=l2_list,
                               preprocess_func=preprocess_func_list,
                               epochs=epochs_list,
                               batch_size=batch_size_list)

    configurations = list(cartesian_product(configuration_space))
    random.shuffle(configurations)

    logger = utils.TSVLogger(save_path) if save_path is not None else None

    for configuration in configurations:
        experiment(train_questions, train_tensor, train_answers,
                   valid_questions, valid_tensor, valid_answers,
                   seed=seed, n_predictors=n_predictors,
                   max_features=max_features, max_len=max_len,
                   logger=logger,
                   **configuration)

    logger.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

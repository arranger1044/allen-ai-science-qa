# -*- coding: utf-8 -*-

import numpy as np

from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.regularizers import l1, l2, l1l2
from keras.initializations import glorot_uniform

import csv
import gensim
import logging


class TSVLogger(object):
    f = None
    writer = None

    def __init__(self, path):
        logging.info('TSVLogger(%s)' % path)
        self.path = path

    def write(self, entry):
        logging.info('TSVLogger.write(%s)' % entry)
        if self.writer is None:
            self.f = open(self.path, 'w')
            self.writer = csv.DictWriter(self.f, entry.keys())
            self.writer.writeheader()
        self.writer.writerow(entry)
        self.f.flush()

    def close(self):
        logging.info('TSVLogger.close()')
        self.f.close()


def compute_accuracy(true_labels, predicted_soft_labels):
    n_observations = true_labels.shape[0]
    predicted_labels = np.argmax(predicted_soft_labels, axis=1)
    assert n_observations == predicted_labels.shape[0]
    return np.sum(true_labels == predicted_labels) / n_observations


def get_optimizer(name, lr=None, momentum=None, decay=None, nesterov=False,
                  epsilon=None, rho=None, beta_1=None, beta_2=None):
    optimizer = None
    if name == 'sgd':
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
    elif name == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=epsilon)
    elif name == 'adadelta':
        optimizer = Adadelta(lr=lr, rho=rho, epsilon=epsilon)
    elif name == 'rmsprop':
        optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
    elif name == 'adam':
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    if optimizer is None:
        raise ValueError('Unknown optimizer: %s' % name)
    return optimizer


def get_recurrent_layer(model_name, input_size, output_size, return_sequences=False):
    layer = None
    if model_name == 'rnn':
        layer = SimpleRNN(input_dim=input_size, output_dim=output_size, return_sequences=return_sequences)
    elif model_name == 'lstm':
        layer = LSTM(input_dim=input_size, output_dim=output_size, return_sequences=return_sequences)
    elif model_name == 'gru':
        layer = GRU(input_dim=input_size, output_dim=output_size, return_sequences=return_sequences)
    if layer is None:
        raise ValueError('Unknown recurrent layer: %s' % model_name)
    return layer


def get_regularizer(lambda_l1=None, lambda_l2=None):
    regularizer = None
    if lambda_l1 is None and lambda_l2 is not None:
        regularizer = l2(l=lambda_l2)
    elif lambda_l1 is not None and lambda_l2 is None:
        regularizer = l1(l=lambda_l1)
    elif lambda_l1 is not None and lambda_l2 is not None:
        regularizer = l1l2(l1=lambda_l1, l2=lambda_l2)
    return regularizer


def read_embeddings(path, token2idx, max_features):
    token2embedding, embedding_size = None, None

    if path.endswith('.txt'):
        with open(path, 'r') as fin:
            token2embedding = dict()
            for line in fin:
                elements = line.split()
                token = elements[0]
                if token in token2idx:
                    embedding_vector = [float(e) for e in elements[1:]]
                    token2embedding[token], embedding_size = embedding_vector, len(embedding_vector)

    elif path.endswith('.model'):
            model = gensim.models.Word2Vec.load(path)
            token2embedding = dict()
            for token in token2idx:
                if token in model:
                    embedding_vector = model[token].tolist()
                    token2embedding[token], embedding_size = embedding_vector, len(embedding_vector)

    else:
        raise ValueError('Unknown format: %s.' % path)

    weights = [get_weights(shape=(max_features, embedding_size), token2idx=token2idx, token2embedding=token2embedding)]
    return weights


def get_weights(shape, token2idx, token2embedding):
    weights = glorot_uniform(shape).get_value()
    if token2embedding is not None:
        vocabulary, tokens = set(token2idx.keys()), set(token2embedding.keys())
        for token_to_initialize in vocabulary.intersection(tokens):
            idx = token2idx[token_to_initialize]
            if idx < weights.shape[0]:
                weights[idx, :] = token2embedding[token_to_initialize]
    return weights

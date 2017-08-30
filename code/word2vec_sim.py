from dataset import load_data_frame
from dataset import get_data_matrix
from dataset import get_answers
from dataset import numbers_to_letters
from dataset import get_ids
from dataset import get_questions_plus_correct_answer
from dataset import save_predictions

from embeddings import word2vec_embedding_model
from embeddings import load_word2vec_model
from embeddings import predict_answers_by_similarity
from embeddings import sum_sentence_embedding, mean_sentence_embedding, prod_sentence_embedding
from embeddings import conv_sentence_embedding
from embeddings import create_bigram_collocations, preprocess_bigram_collocations
from embeddings import predict_answers_by_similarity_neg, predict_answers_by_similarity_llscore
from embeddings import predict_answers_by_similarity_grahm

from preprocess import tokenize_and_lemmatize, tokenize_and_stem, tokenize_lemmatize_and_stem
from preprocess import tokenize_sentence
from preprocess import preprocess_factory
from preprocess import set_stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation import compute_accuracy
from evaluation import cosine_sim, euclidean_sim, manhattan_sim, correlation_sim
from evaluation import create_submission
from evaluation import hard_preds
from evaluation import normalize_preds

from time import perf_counter

import argparse

import logging

import itertools

import os

import datetime

# PREPROCESS_FUNC_DICT = {'t': tokenize_sentence,
#                         'tl': tokenize_and_lemmatize,
#                         'ts': tokenize_and_stem,
#                         'tls': tokenize_lemmatize_and_stem}

RETRIEVE_FUNC_DICT = {'sum': sum_sentence_embedding,
                      'mean': mean_sentence_embedding,
                      'prod': prod_sentence_embedding,
                      'conv': conv_sentence_embedding,
                      }

SIM_FUNC_DICT = {'cos': cosine_sim,
                 'euc': euclidean_sim,
                 'man': manhattan_sim,
                 'cor': correlation_sim}

COS_SIM_FUNCS = {cosine_sim, correlation_sim}

PRED_FUNC_DICT = {'sim': predict_answers_by_similarity,
                  'neg': predict_answers_by_similarity_neg,
                  'log': predict_answers_by_similarity_llscore,
                  'gra': predict_answers_by_similarity_grahm}

PREDS_EXT = '.scores'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str, nargs='+',
                        help='Can be the path to a file or two strings')

    parser.add_argument('-m', '--model', type=str,
                        default='../models/word2vec/051/0/0.model',
                        help='Embedding model file path')

    parser.add_argument('--proc-func',  type=str, nargs='+',
                        default=['tk>rs'],
                        help='Doc preprocessing func pipeline')

    parser.add_argument('--pred-func',  type=str, nargs='+',
                        default=['sim'],
                        help='Predict sim score func')

    parser.add_argument('--ret-func',  type=str, nargs='+',
                        default=['sum'],
                        help='sentence embedding generation func')

    parser.add_argument('--separator',  type=str,
                        default='\t',
                        help='Separator for doc pairs file')

    parser.add_argument('--stopwords',  type=str,
                        default=None,
                        help='Path to stopwords file')

    parser.add_argument('--id',  type=str,
                        default='0',
                        help='Counter to identify the output')

    parser.add_argument('--sim-func',  type=str, nargs='+',
                        default=['cos'],
                        help='similarity function')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/word2vec/',
                        help='Output dir path for predictions file')

    args = parser.parse_args()

    print("Starting with arguments:", args)

    if args.stopwords:
        print('Changing stopwords set')
        stopwords_path = '../stopword/stop-words_english_{}_en.txt'.format(args.stopwords)
        with open(stopwords_path, 'r') as stopwords_file:
            stopwords = [line.strip() for line in stopwords_file.readlines()]
            s = set_stopwords(stopwords)
            print('Set new stopword set ({} words)'.format(len(s)))

    doc_pairs = None

    #
    # two strings passed as input
    if len(args.data) == 2:
        doc_pairs = [args.data]
        assert len(doc_pairs[0]) == 2

        assert isinstance(doc_pairs[0][0], str)
        assert isinstance(doc_pairs[0][1], str)
    #
    # else it is a file path
    elif len(args.data) == 1:
        doc_pairs = []
        data_path = args.data[0]
        assert os.path.isfile(data_path)

        with open(data_path, 'r') as data_pairs:
            for line in data_pairs:
                docs = line.rstrip().split(args.separator)
                doc_pairs.append(docs)
    else:
        raise ValueError('More than two input parameters passed!')

    model_path = args.model
    assert os.path.isfile(model_path)

    model_load_t = perf_counter()
    model = load_word2vec_model(model_path)
    model_end_t = perf_counter()

    embedding_size = model.vector_size
    print('Loaded w2v model {0} in {1} secs, with embedding size: {2}'.format(model_path,
                                                                              model_end_t -
                                                                              model_load_t,
                                                                              embedding_size))
    preprocess_funcs = args.proc_func

    retrieve_funcs = args.ret_func

    sim_funcs = args.sim_func

    pred_funcs = args.pred_func

    params_dict = {'preprocess_func': preprocess_funcs,
                   'retrieve_func': retrieve_funcs,
                   'sim_func': sim_funcs}

    sorted_param_list = sorted(params_dict)

    configurations = None
    configurations = [dict(zip(sorted_param_list, prod))
                      for prod in itertools.product(*(params_dict[param]
                                                      for param in sorted_param_list))]

    for config in configurations:

        retrieve_func = RETRIEVE_FUNC_DICT[config['retrieve_func']]
        sim_func = SIM_FUNC_DICT[config['sim_func']]
        preprocess_func = preprocess_factory(config['preprocess_func'])

        #
        # create output file
        config_param_str = args.id + '.' + \
            '.'.join([str(config[p]) for p in sorted(config)]) + PREDS_EXT
        preds_file_path = os.path.join(args.output, config_param_str)

        print('Processing', config_param_str)
        with open(preds_file_path, 'w') as preds_file:

            for doc_1, doc_2 in doc_pairs:

                #
                # processing docs
                doc_1 = preprocess_func(doc_1)
                doc_2 = preprocess_func(doc_2)

                #
                # create embeddings
                doc_1_emb = retrieve_func(model, doc_1, embedding_size)
                doc_2_emb = retrieve_func(model, doc_2, embedding_size)

                #
                # compute similarity
                doc_sim = sim_func(doc_1_emb, doc_2_emb)

                #
                # store it
                preds_file.write(str(doc_sim) + '\n')

        print('All done.')

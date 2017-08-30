from dataset import load_data_frame
from dataset import get_data_matrix
from dataset import get_answers
from dataset import numbers_to_letters
from dataset import get_ids
from dataset import get_questions_plus_correct_answer

from embeddings import glove_embedding_model
from embeddings import load_glove_model
from embeddings import predict_answers_by_similarity_glove
from embeddings import sum_sentence_embedding, mean_sentence_embedding, prod_sentence_embedding
from embeddings import create_bigram_collocations, preprocess_bigram_collocations
from embeddings import get_glove_embedding, is_word_in_glove_model

from preprocess import tokenize_and_lemmatize, tokenize_and_stem, tokenize_lemmatize_and_stem
from preprocess import tokenize_sentence

from evaluation import compute_accuracy
from evaluation import cosine_sim, euclidean_sim, manhattan_sim, correlation_sim
from evaluation import create_submission

import argparse

import logging

import itertools

import os

import datetime

PREPROCESS_FUNC_DICT = {'t': tokenize_sentence,
                        'tl': tokenize_and_lemmatize,
                        'ts': tokenize_and_stem,
                        'tls': tokenize_lemmatize_and_stem}

RETRIEVE_FUNC_DICT = {'sum': sum_sentence_embedding,
                      'mean': mean_sentence_embedding,
                      'prod': prod_sentence_embedding,
                      }

SIM_FUNC_DICT = {'cos': cosine_sim,
                 'euc': euclidean_sim,
                 'man': manhattan_sim,
                 'cor': correlation_sim}


def glove_eval_train_test(model,
                          train_data,
                          train_target,
                          test_data,
                          test_target,
                          train_ids,
                          test_ids,
                          process_func,
                          ret_func,
                          sim_func,
                          output_path,
                          to_letters=False,
                          bigram_model=None):

    #
    # predict on the training part of questions+answers
    train_questions = train_data[:, 0]
    train_answers = train_data[:, 1:]

    preprocess_func = PREPROCESS_FUNC_DICT[process_func]
    if bigram_model:
        preprocess_func = lambda x: preprocess_bigram_collocations(
            PREPROCESS_FUNC_DICT[process_func](x),
            bigram_model)
    retrieve_func = RETRIEVE_FUNC_DICT[ret_func]
    similarity_func = SIM_FUNC_DICT[sim_func]

    train_predictions = predict_answers_by_similarity_glove(model,
                                                            train_questions,
                                                            train_answers,
                                                            preprocess_func,
                                                            model.no_components,
                                                            to_letters=to_letters,
                                                            retrieve_func=retrieve_func,
                                                            sim_func=similarity_func)

    if train_ids is not None:
        train_pred_file = os.path.join(output_path, 'train.preds')
        create_submission(numbers_to_letters(train_predictions),
                          output=train_pred_file,
                          ids=train_ids)

    train_accuracy = None
    if train_target is not None:
        train_accuracy = compute_accuracy(train_target, train_predictions)

    #
    # now on the test set, if present
    test_accuracy = None
    test_predictions = None
    if test_data is not None:
        test_questions = test_data[:, 0]
        test_answers = test_data[:, 1:]
        test_predictions = predict_answers_by_similarity_glove(model,
                                                               test_questions,
                                                               test_answers,
                                                               preprocess_func,
                                                               model.no_components,
                                                               to_letters=to_letters,
                                                               retrieve_func=retrieve_func,
                                                               sim_func=similarity_func)
        if test_ids is not None:
            test_pred_file = os.path.join(output_path, 'test.preds')
            create_submission(numbers_to_letters(test_predictions),
                              output=test_pred_file,
                              ids=test_ids)

        if test_target is not None:
            test_accuracy = compute_accuracy(test_target, test_predictions)

    print('Train acc {0} test acc {1}'.format(train_accuracy, test_accuracy))

    return train_predictions, test_predictions, train_accuracy, test_accuracy


def glove_train_test(corpus,
                     train_data,
                     train_target,
                     test_data,
                     test_target,
                     train_ids,
                     test_ids,
                     n_jobs,
                     alpha,
                     sample,
                     window_size,
                     embedding_size,
                     skip_gram,
                     cbow_mean,
                     iter,
                     negative_sampling,
                     preprocess_func,
                     retrieve_func,
                     sim_func,
                     output_path,
                     to_letters=False,
                     bigram_model=None,
                     min_count=1):

    #
    # create the model on the training data
    model_name = '-'.join([str(p) for p in [alpha, cbow_mean, embedding_size, iter,
                                            min_count, negative_sampling, sample, skip_gram,
                                            window_size]]) + '.model'
    model_output_path = os.path.join(output_path, model_name)

    model = glove_embedding_model(corpus,
                                  embedding_length=embedding_size,
                                  window_size=window_size,
                                  max_count=min_count,
                                  n_jobs=n_jobs,
                                  alpha=alpha,
                                  sample=sample,
                                  # skip_gram=skip_gram,
                                  # cbow_mean=cbow_mean,
                                  # negative_sampling=negative_sampling,
                                  iter=iter,
                                  # preprocess_func=preprocess_func,
                                  save_path=model_output_path,
                                  trim=False)

    return glove_eval_train_test(model,
                                 train_data,
                                 train_target,
                                 test_data,
                                 test_target,
                                 train_ids,
                                 test_ids,
                                 preprocess_func,
                                 retrieve_func,
                                 sim_func,
                                 output_path,
                                 to_letters=to_letters,
                                 bigram_model=bigram_model)


def parameters_grid_search(model,
                           corpus,
                           train_data,
                           train_target,
                           test_data,
                           test_target,
                           train_ids,
                           test_ids,
                           n_jobs,
                           output_path,
                           params,
                           out_log=None,
                           # save_all_submissions=True,
                           to_letters=False,
                           bigrams=None):

    preamble = 'id\t' + '\t'.join([param for param in sorted(params.keys())]) +\
        '\ttrain-acc\ttest-acc'

    print(preamble)
    out_log.write(preamble + '\n')
    out_log.flush()

    preprocess_funcs = params.pop('preprocess_func')

    # retrieve_funcs = params.pop('retrieve_func')

    # sim_funcs = params.pop('sim_func')

    sorted_param_list = sorted(params)

    configurations = None
    configurations = [dict(zip(sorted_param_list, prod))
                      for prod in itertools.product(*(params[param]
                                                      for param in sorted_param_list))]

    grid_count = 0
    best_acc = 0
    best_config = None
    best_preds = None

    for preprocess in preprocess_funcs:

        #
        # preprocessing corpus
        preprocess_func = PREPROCESS_FUNC_DICT[preprocess]
        if corpus:
            print('Preprocessing corpus')
            processed_corpus = [preprocess_func(d) for d in corpus]

            #
            # collocations?
            if bigrams:
                bigrams = create_bigram_collocations(processed_corpus)
                processed_corpus = [preprocess_bigram_collocations(d, bigrams)
                                    for d in processed_corpus]

        for config in configurations:

            output_exp_path = os.path.join(output_path, str(grid_count))
            os.makedirs(output_exp_path, exist_ok=True)

            if model is None:
                train_preds, test_preds, train_acc, test_acc = \
                    glove_train_test(processed_corpus,
                                     train_data=train_data,
                                     train_target=train_target,
                                     test_data=test_data,
                                     test_target=test_target,
                                     train_ids=train_ids,
                                     test_ids=test_ids,
                                     n_jobs=n_jobs,
                                     preprocess_func=preprocess,
                                     output_path=output_exp_path,
                                     to_letters=to_letters,
                                     bigram_model=bigrams,
                                     # save_all_submissions=save_all_submissions,
                                     **config)

            else:
                train_preds, test_preds, train_acc, test_acc =\
                    glove_eval_train_test(model,
                                          train_data=train_data,
                                          train_target=train_target,
                                          test_data=test_data,
                                          test_target=test_target,
                                          train_ids=train_ids,
                                          test_ids=test_ids,
                                          # config['process_func'],
                                          process_func=preprocess,
                                          ret_func=config['retrieve_func'],
                                          sim_func=config['sim_func'],
                                          output_path=output_exp_path,
                                          to_letters=to_letters)
            if train_acc > best_acc:
                best_acc = train_acc
                best_config = config
                best_preds = test_preds

            update_config = {'preprocess_func': preprocess}
            update_config.update(config)
            param_str = '\t'.join([str(update_config[p]) for p in sorted(update_config)])

            config_str = '\t'.join([str(grid_count),
                                    param_str,
                                    str(train_acc),
                                    str(test_acc)])
            print(config_str)
            out_log.write(config_str + '\n')
            out_log.flush()

            grid_count += 1

    output_best = os.path.join(output_path, 'best.preds')
    create_submission(numbers_to_letters(best_preds), output_best, ids=test_ids)

    print('GRID ENDED')
    print('BEST CONFIG:', best_config)
    print('BEST ACC:', best_acc)

    return best_acc, best_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str,
                        help='Corpus path or model path if evaluating')

    parser.add_argument('-k', '--n-folds', type=int, nargs='?',
                        default=5,
                        help='Number of folds for cv')

    parser.add_argument('--test', type=str,
                        default='../data/validation_set.tsv',
                        help='Test set path')

    parser.add_argument('--train', type=str,
                        default='../data/training_set.tsv',
                        help='Train set path')

    parser.add_argument('-j', '--n-jobs', type=int,
                        default=3,
                        help='Number of jobs for word2vec')

    parser.add_argument('-a', '--alpha',  type=float, nargs='+',
                        default=[0.025],
                        help='Learning rate')

    parser.add_argument('--sample',  type=float, nargs='+',
                        default=[0],
                        help='threshold for configuring which higher - frequency'
                        'words are randomly downsampled')

    parser.add_argument('-w', '--window-size',  type=int, nargs='+',
                        default=[10],
                        help='Window size')

    parser.add_argument('-e', '--embedding-size',  type=int, nargs='+',
                        default=[100],
                        help='Embedding size')

    parser.add_argument('-i', '--iter',  type=int, nargs='+',
                        default=[1],
                        help='Number of passes on dataset')

    parser.add_argument('-s', '--skip-gram',  type=int, nargs='+',
                        default=[1],
                        help='Using skip grams with 1 (default is cbow)')

    parser.add_argument('--min-count',  type=int, nargs='+',
                        default=[1],
                        help='minimum freqs for documents')

    parser.add_argument('--proc-func',  type=str, nargs='+',
                        default=['tl'],
                        help='doc preprocessing func')

    parser.add_argument('--ret-func',  type=str, nargs='+',
                        default=['sum'],
                        help='sentence embedding generation func')

    parser.add_argument('--sim-func',  type=str, nargs='+',
                        default=['cos'],
                        help='similarity function')

    parser.add_argument('--eval', action='store_true',
                        help='minimum freqs for documents')

    parser.add_argument('--use-train', action='store_true',
                        help='minimum freqs for documents')

    parser.add_argument('--cbow-mean', type=int, nargs='+',
                        default=[0],
                        help='when using cbow use the mean instead of the sum')

    parser.add_argument('-n', '--negative-sampling', type=int, nargs='+',
                        default=[0],
                        help='Negative sampling, if 0 hierarchical softmax')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/word2vec/',
                        help='Output dir path')

    args = parser.parse_args()

    print("Starting with arguments:", args)

    #
    # loading the corpus, or the model
    print('Loading', args.data)
    data = None

    #
    # loading the training and validation
    print('Loading training', args.train)
    train_frame = load_data_frame(args.train)

    print('Loading test', args.test)
    test_frame = load_data_frame(args.test)

    #
    # extracting matrices
    train_matrix = get_data_matrix(train_frame)
    train_target = get_answers(train_frame, numeric=True)
    train_ids = get_ids(train_frame)

    test_matrix = get_data_matrix(test_frame)
    test_target = None
    test_ids = get_ids(test_frame)
    try:
        test_target = get_answers(test_frame, numeric=True)
    except:
        pass

    preprocess_funcs = args.proc_func

    retrieve_funcs = args.ret_func

    sim_funcs = args.sim_func

    #
    # opening log file
    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.output + 'exp_' + date_string
    out_log_path = out_path + '/exp.log'

    #
    # # creating dir if non-existant
    if not os.path.exists(os.path.dirname(out_log_path)):
        os.makedirs(os.path.dirname(out_log_path))

    model = None
    corpus = None
    params_dict = None
    #
    # just evaluate?
    if args.eval:

        #
        # data stores the path of a learned model
        print('Loading the model')
        model = load_glove_model(args.data)

        params_dict = {'preprocess_func': preprocess_funcs,
                       'retrieve_func': retrieve_funcs,
                       'sim_func': sim_funcs}

    else:

        with open(args.data, 'r') as corpus_file:
            corpus = corpus_file.readlines()

            #
            # adding training?
            # FIXME: this has to be fixed for a cv, the train score is false
            if args.use_train:
                train_qas = get_questions_plus_correct_answer(train_frame)
                corpus = corpus + train_qas

        #
        # preprocessing once the corpus
        # print('Preprocessing the corpus')
        # preprocess_func = args.corpus_proc_func
        # corpus = [preprocess_func(d) for d in corpus]

        #
        # creating params dictionary
        params_dict = {'embedding_size': args.embedding_size,
                       'window_size': args.window_size,
                       'min_count': args.min_count,
                       'alpha': args.alpha,
                       'sample': args.sample,
                       # 'skip_gram': args.skip_gram,
                       # 'cbow_mean': args.cbow_mean,
                       'iter': args.iter,
                       # 'negative_sampling': args.negative_sampling,
                       'preprocess_func': preprocess_funcs,
                       'retrieve_func': retrieve_funcs,
                       'sim_func': sim_funcs}

    with open(out_log_path, 'w') as out_log:

        parameters_grid_search(model,
                               corpus,
                               train_matrix,
                               train_target,
                               test_matrix,
                               test_target,
                               train_ids,
                               test_ids,
                               n_jobs=args.n_jobs,
                               output_path=out_path,
                               params=params_dict,
                               out_log=out_log,
                               to_letters=False)

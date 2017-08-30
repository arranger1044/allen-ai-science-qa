from dataset import load_data_frame
from dataset import get_data_matrix, get_ids, get_questions, get_answers

from preprocess import tokenize_and_lemmatize, tokenize_and_stem, tokenize_lemmatize_and_stem
from preprocess import tokenize_sentence
from preprocess import preprocess_factory

import argparse

from collections import defaultdict

from time import perf_counter

import logging

import os

N_ANSWERS = 4


def count_and_report(matrix, questions, answers, ids, vocabulary, verbose):
    INVERSE_ID_DICT = {}
    EMPTY_QUESTIONS = set()
    EMPTY_ANSWERS = defaultdict(set)
    MISSING_TOKENS = set()
    QUESTIONS_WITH_EMPTY_ANSWERS = defaultdict(set)  # {i: set for i in range(N_ANSWERS)}
    QUESTIONS_WITH_MISSING_TOKENS = defaultdict(set)

    for i in range(matrix.shape[0]):

        INVERSE_ID_DICT[ids[i]] = i
        q = questions[i]
        q_len = 0
        miss_q = 0
        for w in q:
            if w in vocabulary:
                q_len += 1
            else:
                MISSING_TOKENS.add(w)
                QUESTIONS_WITH_MISSING_TOKENS[miss_q].add(ids[i])
                miss_q += 1
        if q_len == 0:
            EMPTY_QUESTIONS.add(ids[i])

        ans = answers[i]
        empty_ans = 0
        for a in ans:
            a_len = 0
            for w in a:

                if w in vocabulary:
                    a_len += 1
                else:
                    MISSING_TOKENS.add(w)
            if a_len == 0:
                EMPTY_ANSWERS[ids[i]].add(tuple(a))
                QUESTIONS_WITH_EMPTY_ANSWERS[empty_ans].add(ids[i])
                empty_ans += 1

    print('\t# missing tokens: \t{}'.format(len(MISSING_TOKENS)))

    if verbose > 0:
        print(sorted(MISSING_TOKENS))

    print('\t# empty questions:\t{}'.format(len(EMPTY_QUESTIONS)))
    if verbose > 0:
        print('\t\tEmpty questions')
        if verbose == 1:
            print_questions = [id for id in sorted(EMPTY_QUESTIONS)]
        elif verbose == 2:
            print_questions = '\n'.join(['{0}: {1}'.format(id, questions[INVERSE_ID_DICT[id]])
                                         for id in sorted(EMPTY_QUESTIONS)])
        print(print_questions)

    print('\t# empty answers:\t{}'.format(len(EMPTY_ANSWERS)))
    if verbose > 0:
        print('\t\tEmpty answers')
        if verbose == 1:
            print_answers = [id for id in sorted(EMPTY_ANSWERS.keys())]
        elif verbose == 2:
            print_answers = '\n'.join(['{0}: {1}'.format(id, EMPTY_ANSWERS[id])
                                       for id in sorted(EMPTY_ANSWERS.keys())])

        print(print_answers)

    for i in range(4):
        print('\t# questions with at least {0} empty answers:\t{1}'.format(
            i + 1, len(QUESTIONS_WITH_EMPTY_ANSWERS[i])))
        if verbose > 0:
            if verbose == 1:
                print_questions = [id for id in sorted(QUESTIONS_WITH_EMPTY_ANSWERS[i])]
            elif verbose == 2:
                print_questions = '\n'.join(['{0}: {1}'.format(id,
                                                               questions[INVERSE_ID_DICT[QUESTIONS_WITH_EMPTY_ANSWERS[i][id]]])
                                             for id in sorted(QUESTIONS_WITH_EMPTY_ANSWERS[i])])
            print(print_questions)

    for i in sorted(QUESTIONS_WITH_MISSING_TOKENS.keys()):
        print('\t# questions with at leat {0} missing tokens:\t{1}'.format(i + 1,
                                                                           len(QUESTIONS_WITH_MISSING_TOKENS[i])))
        if verbose > 0:

            if verbose == 1:
                print_questions = [id for id in sorted(QUESTIONS_WITH_MISSING_TOKENS[i])]
            elif verbose == 2:
                print_questions = '\n'.join(['{0}: {1}'.format(id,
                                                               questions[INVERSE_ID_DICT[QUESTIONS_WITH_MISSING_TOKENS[i][id]]])
                                             for id in sorted(QUESTIONS_WITH_MISSING_TOKENS[i])])
            print(print_questions)


# PREPROCESS_FUNC_DICT = {'t': tokenize_sentence,
#                         'tl': tokenize_and_lemmatize,
#                         'ts': tokenize_and_stem,
#                         'tls': tokenize_lemmatize_and_stem}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('corpus', type=str,
                        help='Corpus path or model path if evaluating')

    parser.add_argument('--test', type=str,
                        default='../data/validation_set.tsv',
                        help='Test set path')

    parser.add_argument('--train', type=str,
                        default='../data/training_set.tsv',
                        help='Train set path')

    parser.add_argument('--proc-func',  type=str,
                        default='tk>rs',
                        help='doc preprocessing func')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/word2vec/',
                        help='Output dir path')

    parser.add_argument('-v', '--verbose', type=int,
                        default=0,
                        help='Verbosity level in 0, 1, 2')

    args = parser.parse_args()

    print("Starting with arguments:", args)

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
    train_questions = get_questions(train_frame)

    test_matrix = get_data_matrix(test_frame)
    test_questions = get_questions(test_frame)
    test_target = None
    test_ids = get_ids(test_frame)
    try:
        test_target = get_answers(test_frame, numeric=True)
    except:
        pass

    # preprocess_func = PREPROCESS_FUNC_DICT[args.proc_func]
    preprocess_func = preprocess_factory(args.proc_func)

    with open(args.corpus, 'r') as corpus_file:
        corpus = corpus_file.readlines()

    #
    #
    print('Preprocessing corpus')
    proc_start_t = perf_counter()
    corpus = [preprocess_func(d) for d in corpus]
    proc_end_t = perf_counter()
    print('done in {} secs'.format(proc_end_t - proc_start_t))

    print('Creating corpus vocabulary')
    proc_start_t = perf_counter()
    vocabulary = set(t for d in corpus for t in d)
    proc_end_t = perf_counter()
    print('done in {} secs'.format(proc_end_t - proc_start_t))
    print('There are {} words'.format(len(vocabulary)))

    print('\nPreprocessing training')
    proc_start_t = perf_counter()
    for i in range(train_matrix.shape[0]):
        for j in range(train_matrix.shape[1]):
            train_matrix[i, j] = preprocess_func(train_matrix[i, j])
    proc_end_t = perf_counter()
    print('done in {} secs'.format(proc_end_t - proc_start_t))

    train_questions = train_matrix[:, 0]
    train_answers = train_matrix[:, 1:]

    print('\nPreprocessing validation')
    proc_start_t = perf_counter()
    for i in range(test_matrix.shape[0]):
        for j in range(test_matrix.shape[1]):
            test_matrix[i, j] = preprocess_func(test_matrix[i, j])
    proc_end_t = perf_counter()
    print('done in {} secs'.format(proc_end_t - proc_start_t))

    test_questions = test_matrix[:, 0]
    test_answers = test_matrix[:, 1:]

    print('\n\nTrain reporting:')
    count_and_report(
        train_matrix, train_questions, train_answers, train_ids, vocabulary, args.verbose)

    print('\n\nValidation reporting:')
    count_and_report(
        test_matrix, test_questions, test_answers, test_ids, vocabulary, args.verbose)

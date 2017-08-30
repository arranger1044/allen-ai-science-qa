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
from embeddings import predict_answers_by_similarity_avg

# from preprocess import tokenize_and_lemmatize, tokenize_and_stem, tokenize_lemmatize_and_stem
# from preprocess import tokenize_sentence
from preprocess import preprocess_factory
from preprocess import set_stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation import compute_accuracy
from evaluation import cosine_sim, euclidean_sim, manhattan_sim, correlation_sim
from evaluation import create_submission
from evaluation import hard_preds
from evaluation import normalize_preds

import argparse

import logging

import itertools

import os

import datetime

from time import perf_counter

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

# COS_SIM_FUNCS = {cosine_sim, correlation_sim}
COS_SIM_FUNCS = {'cos',  'cor'}


PRED_FUNC_DICT = {'sim': predict_answers_by_similarity,
                  'neg': predict_answers_by_similarity_neg,
                  'log': predict_answers_by_similarity_llscore,
                  'gra': predict_answers_by_similarity_grahm,
                  'avg': predict_answers_by_similarity}


def word2vec_eval_train_test(model,
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
                             pred_func,
                             tfidf_weight=None,
                             to_letters=False,
                             normalize=True,
                             prefix_str="",
                             bigram_model=None):

    #
    # predict on the training part of questions+answers
    train_questions = train_data[:, 0]
    train_answers = train_data[:, 1:]

    # preprocess_func = PREPROCESS_FUNC_DICT[process_func]
    preprocess_func = preprocess_factory(process_func)
    if bigram_model:
        preprocess_func = lambda x: preprocess_bigram_collocations(
            # PREPROCESS_FUNC_DICT[process_func](x),
            preprocess_factory(process_func)(x),
            bigram_model)
    retrieve_func = RETRIEVE_FUNC_DICT[ret_func]
    similarity_func = SIM_FUNC_DICT[sim_func]

    cos_sim = False
    # if retrieve_func in COS_SIM_FUNCS:
    if sim_func in COS_SIM_FUNCS:
        cos_sim = True

    prediction_func = PRED_FUNC_DICT[pred_func]

    train_predictions = prediction_func(model,
                                        train_questions,
                                        train_answers,
                                        preprocess_func,
                                        model.vector_size,
                                        # to_letters=to_letters,
                                        retrieve_func=retrieve_func,
                                        sim_func=similarity_func,
                                        tfidf_weight=tfidf_weight)

    if normalize:
        train_predictions = normalize_preds(train_predictions, cos_sim)

    #
    # save soft predictions
    train_pred_file = os.path.join(output_path, prefix_str + 'train.scores')
    save_predictions(train_predictions, train_pred_file, ids=train_ids)

    #
    # hard predictions
    if train_ids is not None:
        hard_train_predictions = hard_preds(train_predictions)
        train_pred_file = os.path.join(output_path, prefix_str + 'train.submission')
        create_submission(numbers_to_letters(hard_train_predictions),
                          output=train_pred_file,
                          ids=train_ids)

    train_accuracy = None
    if train_target is not None:
        train_accuracy = compute_accuracy(train_target, hard_train_predictions)

    #
    # now on the test set, if present
    test_accuracy = None
    test_predictions = None
    if test_data is not None:
        test_questions = test_data[:, 0]
        test_answers = test_data[:, 1:]
        test_predictions = prediction_func(model,
                                           test_questions,
                                           test_answers,
                                           preprocess_func,
                                           model.vector_size,
                                           # to_letters=to_letters,
                                           retrieve_func=retrieve_func,
                                           sim_func=similarity_func,
                                           tfidf_weight=tfidf_weight)

        if normalize:
            test_predictions = normalize_preds(test_predictions, cos_sim)

        #
        # save soft predictions
        test_pred_file = os.path.join(output_path, 'valid.scores')
        save_predictions(test_predictions, test_pred_file, ids=test_ids)
        #
        # hard predictions
        if test_ids is not None:
            hard_test_predictions = hard_preds(test_predictions)
            test_pred_file = os.path.join(output_path, 'valid.submission')
            create_submission(numbers_to_letters(hard_test_predictions),
                              output=test_pred_file,
                              ids=test_ids)

        if test_target is not None:
            test_accuracy = compute_accuracy(test_target, hard_test_predictions)

    print('Train acc {0} test acc {1}'.format(train_accuracy, test_accuracy))

    return train_predictions, test_predictions, train_accuracy, test_accuracy


def word2vec_train_test(corpus,
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
                        pred_func=predict_answers_by_similarity,
                        to_letters=False,
                        bigram_model=None,
                        c_save_format=True,
                        min_count=1):

    #
    # create the model on the training data
    model_name = '-'.join([str(p) for p in [alpha, cbow_mean, embedding_size, iter,
                                            min_count, negative_sampling, sample, skip_gram,
                                            window_size]]) + '.model'
    model_output_path = os.path.join(output_path, model_name)

    model = word2vec_embedding_model(corpus,
                                     embedding_length=embedding_size,
                                     window_size=window_size,
                                     min_count=min_count,
                                     n_jobs=n_jobs,
                                     alpha=alpha,
                                     sample=sample,
                                     skip_gram=skip_gram,
                                     cbow_mean=cbow_mean,
                                     negative_sampling=negative_sampling,
                                     iter=iter,
                                     # preprocess_func=preprocess_func,
                                     save_path=model_output_path,
                                     c_save_format=c_save_format,
                                     trim=False)

    return word2vec_eval_train_test(model,
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
                                    pred_func=pred_func,
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
                           c_save_format=False,
                           bigrams=None):

    only_eval = False
    if model is not None:
        only_eval = True

    preamble = 'id\t' + '\t'.join([param for param in sorted(params.keys())]) +\
        '\ttrain-acc\ttest-acc'

    print(preamble)
    out_log.write(preamble + '\n')
    out_log.flush()

    preprocess_funcs = params.pop('preprocess_func')
    retrieve_funcs = params.pop('retrieve_func')
    sim_funcs = params.pop('sim_func')
    pred_funcs = params.pop('pred_func')

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
        # preprocess_func = PREPROCESS_FUNC_DICT[preprocess]
        preprocess_func = preprocess_factory(preprocess)
        if corpus:
            print('Preprocessing corpus...', end=' ')
            prep_start_t = perf_counter()
            processed_corpus = [preprocess_func(d) for d in corpus]
            prep_end_t = perf_counter()
            print('done in ', prep_end_t - prep_start_t)

        #
        # collocations?
            if bigrams:
                bigrams = create_bigram_collocations(processed_corpus)
                processed_corpus = [preprocess_bigram_collocations(d, bigrams)
                                    for d in processed_corpus]

        for config in configurations:

            tfidf_dict = None

            output_exp_path = os.path.join(output_path, str(grid_count))
            os.makedirs(output_exp_path, exist_ok=True)

            # if model is None:
            if not only_eval:

                if config['tfidf_weight']:

                    print('Computing Tf-idf weighting')

                    corpus_voc = set()
                    tfidf_corpus = []
                    for s in processed_corpus:
                        corpus_line = ' '.join(s)
                        tfidf_corpus.append(corpus_line)
                        for w in s:
                            corpus_voc.add(w)
                    vectorizer = TfidfVectorizer(min_df=1, vocabulary=corpus_voc)
                    vectorizer.fit(tfidf_corpus)
                    tfidf_dict = dict(zip(vectorizer.get_feature_names(),
                                          vectorizer.idf_))

                model_name = '{}.model'.format(grid_count)
                model_output_path = os.path.join(output_exp_path, model_name)

                config.pop('tfidf_weight')
                model = word2vec_embedding_model(processed_corpus,
                                                 n_jobs=n_jobs,
                                                 # iter=iter,
                                                 # preprocess_func=preprocess_func,
                                                 save_path=model_output_path,
                                                 trim=False,
                                                 c_text_format=c_save_format,
                                                 **config)

            for ret_func, sim_func, pred_func in itertools.product(retrieve_funcs,
                                                                   sim_funcs,
                                                                   pred_funcs):

                prefix_str = 'w2v_{0}_{1}_{2}_{3}_{4}_{5}_{6}.'.format(config['alpha'],
                                                                       config['embedding_size'],
                                                                       config['iter'],
                                                                       config['window_size'],
                                                                       ret_func,
                                                                       sim_func,
                                                                       pred_func
                                                                       )
                output_exp_path = os.path.join(output_path, str(grid_count))
                os.makedirs(output_exp_path, exist_ok=True)

                # else:
                train_preds, test_preds, train_acc, test_acc =\
                    word2vec_eval_train_test(model,
                                             train_data=train_data,
                                             train_target=train_target,
                                             test_data=test_data,
                                             test_target=test_target,
                                             train_ids=train_ids,
                                             test_ids=test_ids,
                                             # config['process_func'],
                                             process_func=preprocess,
                                             ret_func=ret_func,
                                             sim_func=sim_func,
                                             pred_func=pred_func,
                                             tfidf_weight=tfidf_dict,
                                             output_path=output_exp_path,
                                             normalize=config['normalize'],
                                             prefix_str=prefix_str,
                                             to_letters=to_letters)
                if train_acc > best_acc:
                    best_acc = train_acc
                    best_config = config
                    best_preds = test_preds

                update_config = {'preprocess_func': preprocess,
                                 'retrieve_func': ret_func,
                                 'sim_func': sim_func,
                                 'pred_func': pred_func}
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
    save_predictions(best_preds, output_best)

    hard_best_preds = hard_preds(best_preds)
    output_best = os.path.join(output_path, 'best.submission')
    create_submission(numbers_to_letters(hard_best_preds), output_best, ids=test_ids)

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
                        default=['tk>rs'],
                        help='doc preprocessing func')

    parser.add_argument('--pred-func',  type=str, nargs='+',
                        default=['sim'],
                        help='Predict sim score func')

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

    parser.add_argument('--tfidf-weight', nargs='+',
                        type=int,
                        default=[0],
                        help='weighting by tfidf')

    parser.add_argument('--normalize', nargs='+',
                        type=int,
                        default=[1],
                        help='normalize preds in 0 1')

    parser.add_argument('--stopwords',  type=str,
                        default=None,
                        help='Path to stopwords file')

    parser.add_argument('--cbow-mean', type=int, nargs='+',
                        default=[0],
                        help='when using cbow use the mean instead of the sum')

    parser.add_argument('--seed', type=int, nargs='+',
                        default=[1234],
                        help='Seeding for random number generation')

    parser.add_argument('-n', '--negative-sampling', type=int, nargs='+',
                        default=[0],
                        help='Negative sampling, if 0 hierarchical softmax')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='../models/word2vec/',
                        help='Output dir path')

    parser.add_argument('--exp-name', type=str, nargs='?',
                        default=None,
                        help='Experiment name, if not present a date will be used')

    args = parser.parse_args()

    print("Starting with arguments:", args)

    #
    # loading the corpus, or the model
    print('Loading', args.data)
    data = None

    if args.stopwords:
        print('Changing stopwords set')
        stopwords_path = '../stopword/stop-words_english_{}_en.txt'.format(args.stopwords)
        with open(stopwords_path, 'r') as stopwords_file:
            stopwords = [line.strip() for line in stopwords_file.readlines()]
            s = set_stopwords(stopwords)
            print('Set new stopword set ({} words)'.format(len(s)))

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

    tfidf_weights = []
    for w in args.tfidf_weight:
        tw = True if w == 1 else False
        tfidf_weights.append(tw)

    normalize = []
    for n in args.normalize:
        tn = True if n == 1 else False
        normalize.append(tn)

    preprocess_funcs = args.proc_func

    retrieve_funcs = args.ret_func

    sim_funcs = args.sim_func

    pred_funcs = args.pred_func

    #
    # opening log file
    logging.info('Opening log file...')
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = None
    if args.exp_name:
        out_path = os.path.join(args.output, 'exp_' + args.exp_name)
    else:
        out_path = os.path.join(args.output, 'exp_' + date_string)
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
        model = load_word2vec_model(args.data)

        params_dict = {'alpha': args.alpha,
                       'preprocess_func': preprocess_funcs,
                       'retrieve_func': retrieve_funcs,
                       'sim_func': sim_funcs,
                       'embedding_size': args.embedding_size,
                       'normalize': normalize,
                       'iter': args.iter,
                       'window_size': args.window_size,
                       'pred_func': pred_funcs}

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
                       'skip_gram': args.skip_gram,
                       'cbow_mean': args.cbow_mean,
                       'iter': args.iter,
                       'negative_sampling': args.negative_sampling,
                       'preprocess_func': preprocess_funcs,
                       'retrieve_func': retrieve_funcs,
                       'sim_func': sim_funcs,
                       'tfidf_weight': tfidf_weights,
                       'seed': args.seed,
                       'normalize': normalize,
                       'pred_func': pred_funcs}

    with open(out_log_path, 'w') as out_log:

        out_log.write(str(args) + '\n')

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

import numpy

import scipy


import gensim

# from preprocess import tokenize_corpus
# from preprocess import lemmatize_tokens
# from preprocess import stem_tokens
# from preprocess import tokenize_and_lemmatize

from dataset import numbers_to_letters
from dataset import get_questions_plus_answers_labelled
from dataset import get_questions_plus_answers_matrix

from evaluation import cosine_sim
from evaluation import hard_preds
from evaluation import normalize_preds

from preprocess import get_common_tokens, get_token_residual, remove_common_tokens

from time import perf_counter

import itertools

# from glove import Glove
# from glove import Corpus

N_JOBS = 3


def gram_schmidt_ortho(X, row_vecs=False, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = numpy.diag((X[i, :].dot(Y.T) / numpy.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = numpy.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = numpy.diag(1 / numpy.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


def gram_schmidt_ortho_2(A):
    # print(A)
    Q = numpy.zeros(A.shape)

    for k in range(A.shape[1]):
        q = A[:, k]
        for j in range(k):
            q = q - numpy.dot(q, Q[:, j]) * Q[:, j]

        # print(q, numpy.linalg.norm(q))
        Q[:, k] = q / numpy.linalg.norm(q)

    return Q


def gram_schmidt_ortho_piero(A):

    a = A[0]
    s = numpy.zeros(a.shape[0])
    for b in A[1:]:
        s += numpy.dot(a, b) * b

    return a - s


def gram_schmidt(x, y):
    assert len(x) == len(y)
    X = numpy.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    # print(X.shape)
    Y = gram_schmidt_ortho_2(X)
    # print(Y[:, 1])
    return Y[:, 1]


def create_bigram_collocations(sentence):
    return gensim.models.Phrases(sentence, min_count=10, threshold=20)


def preprocess_bigram_collocations(sentence, model):
    return model[sentence]


# def glove_embedding_model(corpus,
#                           embedding_length=100,
#                           window_size=5,
#                           max_count=100,
#                           learning_rate=0.05,
#                           alpha=0.75,
#                           max_loss=10.0,
#                           n_jobs=N_JOBS,
#                           save_path=None,
#                           iter=1,
#                           seed=1234,
#                           verbose=False):

#     #
#     # create the corpus
#     print('Creating the corpus')
#     corpus_model = Corpus()
#     corpus_model.fit(corpus, window=window_size)
#     # corpus_model.save('corpus.model')

#     print('Dict size: %s' % len(corpus_model.dictionary))
#     print('Collocations: %s' % corpus_model.matrix.nnz)

#     #
#     # fit the model
#     glove_model = Glove(no_components=embedding_length,
#                         learning_rate=learning_rate,
#                         alpha=alpha,
#                         max_count=max_count,
#                         max_loss=max_loss)
#     glove_model.fit(corpus_model.matrix, epochs=iter,
#                     no_threads=n_jobs,
#                     verbose=verbose)
#     glove_model.add_dictionary(corpus_model.dictionary)

#     if save_path:
#         glove_model.save(save_path)

#     return glove_model


def get_glove_embedding(model, word):
    return model.word_vectors[model.dictionary[word]]


def is_word_in_glove_model(model, word):
    return word in model.dictionary


# def load_glove_model(model_path):
#     return Glove.load(model_path)


def word2vec_embedding_model(corpus,
                             embedding_size=100,
                             window_size=5,
                             min_count=1,
                             n_jobs=N_JOBS,
                             alpha=0.025,
                             sample=0.0,
                             skip_gram=0,
                             cbow_mean=0,
                             negative_sampling=0,
                             # preprocess_func=tokenize_and_lemmatize,
                             save_path=None,
                             iter=1,
                             trim=False,
                             normalize=None,
                             c_text_format=True,
                             seed=1234):

    # print(embedding_length, window_size,
    #       min_count, n_jobs, alpha, sample, skip_gram, cbow_mean, negative_sampling, save_path)
    model_start_t = perf_counter()
    model = gensim.models.Word2Vec(corpus,
                                   size=embedding_size,
                                   window=window_size,
                                   min_count=min_count,
                                   workers=n_jobs,
                                   alpha=alpha,
                                   sample=sample,
                                   sg=skip_gram,
                                   iter=iter,
                                   cbow_mean=cbow_mean,
                                   negative=negative_sampling,
                                   seed=seed)

    model_end_t = perf_counter()
    print('word2vec learned in {0} secs'.format(model_end_t - model_start_t))

    #
    # save memory
    if trim:
        model.init_sims(replace=True)
    #
    # saving the model
    if save_path:
        if c_text_format:
            model.save_word2vec_format(save_path)
        else:
            model.save(save_path)

    return model


def doc2vec_embedding_model(corpus,
                            embedding_size=100,
                            window_size=5,
                            min_count=1,
                            n_jobs=N_JOBS,
                            alpha=0.025,
                            sample=0.0,
                            dist_mem=1,
                            dbow_words=0,
                            negative_sampling=0,
                            dm_mean=0,
                            dm_concat=0,
                            dm_tag_count=1,
                            docvecs=None,
                            docvecs_mapfile=None,
                            # preprocess_func=tokenize_and_lemmatize,
                            save_path=None,
                            iter=1,
                            trim=False,
                            seed=1234):

    tagged_corpus = [gensim.models.doc2vec.TaggedDocument(words=d,
                                                          tags=['s-{}'.format(i)])
                     for i, d in enumerate(corpus)]
    #
    #

    # print(embedding_length, window_size,
    #       min_count, n_jobs, alpha, sample, skip_gram, cbow_mean, negative_sampling, save_path)
    model_start_t = perf_counter()
    model = gensim.models.Doc2Vec(documents=tagged_corpus,
                                  size=embedding_size,
                                  window=window_size,
                                  min_count=min_count,
                                  workers=n_jobs,
                                  alpha=alpha,
                                  sample=sample,
                                  dbow_words=dbow_words,
                                  dm_mean=dm_mean,
                                  dm_concat=dm_concat,
                                  dm_tag_count=dm_tag_count,
                                  docvecs=docvecs,
                                  docvecs_mapfile=docvecs_mapfile,
                                  dm=dist_mem,
                                  iter=iter,
                                  negative=negative_sampling,
                                  seed=seed)

    model_end_t = perf_counter()
    print('doc2vec learned in {0} secs'.format(model_end_t - model_start_t))

    #
    # save memory
    if trim:
        model.init_sims(replace=True)
    #
    # saving the model
    if save_path:
        model.save(save_path)

    return model


def load_word2vec_model(path, binary=True):
    return gensim.models.Word2Vec.load(path)
    # return gensim.models.Word2Vec.load_word2vec_format(path)


def load_doc2vec_model(path, binary=True):
    return gensim.models.Doc2Vec.load(path)


def load_glove_model_l(path):

    model = {}
    with open(path, 'r') as model_file:
        for line in model_file:
            parsed_line = line.strip().split()
            model[parsed_line[0]] = numpy.array([float(n) for n in parsed_line[1:]])

    return model


def ll_sentence_embedding(model, sentence, n_epochs=50, ignore_missing=True):

    return model.transform_paragraph(sentence, epochs=n_epochs,
                                     ignore_missing=ignore_missing)


def get_model_embedding_size(model, model_type):
    if model_type == 'word2vec':
        return model.vector_size
    elif model_type == 'glove':
        return model.no_components

EPSILON = 1e-10


def zero_embedding(embedding_length, epsilon=EPSILON):
    embedding = numpy.zeros(embedding_length) + EPSILON
    return embedding


def sum_sentence_embedding(model, sentence, embedding_length, tfidf_weight=None):
    """
    Simply returning a sum for the embedding of the words recognized in the model
    """
    embedding = zero_embedding(embedding_length)  # numpy.zeros(embedding_length)
    for word in sentence:
        if word in model:
            weight = 1.0
            if tfidf_weight:
                weight = tfidf_weight[word]
            embedding += weight * model[word]

    return embedding


def infer_sentence_embedding(model, sentence, embedding_length, tfidf_weight=None,
                             alpha=0.025, min_alpha=0.0001, steps=50):
    return model.infer_vector(sentence, alpha=alpha, min_alpha=min_alpha, steps=steps)


def sum_sentence_embedding_glove(model, sentence, embedding_length):
    """
    Simply returning a sum for the embedding of the words recognized in the model
    """
    embedding = zero_embedding(embedding_length)  # numpy.zeros(embedding_length)
    for word in sentence:
        if is_word_in_glove_model(model, word):
            embedding += get_glove_embedding(model, word)

    return embedding


def mean_sentence_embedding(model, sentence, embedding_length):
    """
    Returning a mean for the embedding of the words recognized in the model
    """
    embedding = sum_sentence_embedding(model, sentence, embedding_length)
    n_words = len(sentence)

    return embedding / n_words


def prod_sentence_embedding_old(model, sentence, embedding_length, i=10):
    """
    Simply returning a sum for the embedding of the words recognized in the model
    """
    words_in_model = [word for word in sentence if word in model]

    if not words_in_model:
        if i < 10:
            print(words_in_model)
        return zero_embedding(embedding_length)  # numpy.zeros(embedding_length)

    embedding = model[words_in_model[0]]
    for word in words_in_model[1:]:
        # if word in model:
        embedding *= model[word]
    # print(embedding)
    return embedding


def prod_sentence_embedding_sigmoid(model, sentence, embedding_length):
    """
    Simply returning a sum for the embedding of the words recognized in the model
    """
    words_in_model = [word for word in sentence if word in model]

    if not words_in_model:

        return zero_embedding(embedding_length)  # numpy.zeros(embedding_length)

    embedding = zero_embedding(embedding_length)
    for word in words_in_model:
        # if word in model:
        embedding += numpy.log(scipy.special.expit(model[word]))
    # print(embedding)
    return numpy.exp(embedding)


def prod_sentence_embedding(model, sentence, embedding_length):
    """
    Simply returning a sum for the embedding of the words recognized in the model
    """
    words_in_model = [word for word in sentence if word in model]

    if not words_in_model:

        return zero_embedding(embedding_length)  # numpy.zeros(embedding_length)

    words_embedding = numpy.array([model[word] for word in words_in_model])
    #
    # getting the offset to translate
    offset = words_embedding.min()
    words_embedding += -offset
    # print(words_embedding)

    embedding = numpy.log(words_embedding)
    # embedding = words_embedding[0]
    # for word_emb in words_embedding[1:]:
    #     # if word in model:
    #     embedding += numpy.log(words_embedding[i])
    # # print(embedding)
    # print(embedding)
    return numpy.exp(numpy.sum(embedding, axis=0))  # + offset


def circular_convolution(x, y):
    n = x.shape[0]

    assert y.shape[0] == n

    # x = numpy.log(x)
    # y = numpy.log(y)

    conv = numpy.zeros(n) + EPSILON

    for i in range(n):
        for j in range(n):
            conv[i] += numpy.exp(x[i] + y[(i - j) % n])

    return conv


def conv_sentence_embedding_old(model, sentence, embedding_length):
    """
    Convolving the embeddings
    """

    embeddings = [model[word] for word in sentence if word in model]

    if len(embeddings) < 1:
        return zero_embedding(embedding_length)

    embedding = embeddings[0]
    for w_emb in embeddings[1:]:
        # embedding = numpy.convolve(embedding, w_emb, mode='samex\')
        # embedding = scipy.ndimage.convolve(numpy.log(scipy.special.expit(embedding)),
        #                                    numpy.log(scipy.special.expit(w_emb)), mode='wrap')
        embedding = circular_convolution(embedding, w_emb)

    return embedding


def conv_sentence_embedding(model, sentence, embedding_length):
    """
    Convolving the embeddings
    """

    embeddings = numpy.array([model[word] for word in sentence if word in model])

    if len(embeddings) < 1:
        return zero_embedding(embedding_length)

    #
    # getting the offset to translate
    offset = embeddings.min()
    embeddings += -offset
    # print(words_embedding)

    embeddings = numpy.log(embeddings)

    conv = numpy.zeros((embeddings.shape[0] - 1,
                        embeddings.shape[1]))

    for i in range(len(embeddings) - 1):
        # conv[i] = embeddings[i] + embeddings[i + 1]
        conv[i] = circular_convolution(embeddings[i],
                                       embeddings[i + 1])

    return numpy.sum(conv[:len(embeddings) - 1], axis=0)


def predict_answers_by_similarity(model,
                                  questions,
                                  answers_list,
                                  preprocess_func,
                                  embedding_length,
                                  hard_predictions=False,
                                  # to_letters=False,
                                  # normalize=True,
                                  retrieve_func=sum_sentence_embedding,
                                  sim_func=cosine_sim,
                                  tfidf_weight=None):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        #
        # get and process question
        question = preprocess_func(questions[i])
        question_emb = retrieve_func(model, question, embedding_length, tfidf_weight)

        answers = answers_list[i]

        # best_answer = 0  # default to the first one
        # best_sim = -numpy.inf

        for j, answer in enumerate(answers):

            answer = preprocess_func(answer)
            answer_emb = retrieve_func(model, answer, embedding_length, tfidf_weight)

            qa_sim = sim_func(question_emb, answer_emb)
            predictions[i, j] = qa_sim
            # if qa_sim > best_sim:
            #     best_sim = qa_sim
            #     best_answer = j

        # predictions[i] = best_answer

    # if normalize:
    #     predictions = normalize_preds(predictions)
    # if to_letters:
    #     predictions = numbers_to_letters(predictions)

    return predictions


def predict_answers_by_similarity_neg(model,
                                      questions,
                                      answers_list,
                                      preprocess_func,
                                      embedding_length,
                                      # to_letters=True,
                                      # normalize=True,
                                      retrieve_func=sum_sentence_embedding,
                                      sim_func=cosine_sim,
                                      tfidf_weight=None):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        #
        # get and process question
        question = preprocess_func(questions[i])
        question_emb = retrieve_func(model, question, embedding_length, tfidf_weight)

        # if i < 5:
        #     print(retrieve_func)
        #     print(question)
        #     print(question_emb)

        answers = answers_list[i]

        best_answer = 0  # default to the first one
        best_sim = -numpy.inf

        # common_part_answers = get_common_tokens(answers)
        # n_answers = len(answers)
        # answers_embeddings = []

        # residuals = []

        # for j, answer in enumerate(answers):
        #     answer = preprocess_func(answer)
        #     residuals.append(retrieve_func(model,
        #                                    get_token_residual(answer, common_part_answers),
        #                                    embedding_length))

        #
        # emb_res(A) -(emb_res(B) + emb_res(C) + emb_res(D))

        processed_answers = [preprocess_func(a) for a in answers]

        for j, answer in enumerate(answers):
            # answer = preprocess_func(answer)
            # other_answers = [preprocess_func(answers[k]) for k in range(n_answers) if k != j]
            answer = processed_answers[j]
            other_answers = [p for k, p in enumerate(processed_answers) if k != j]

            other_answers = remove_common_tokens(answer, other_answers)

            other_answers_emb = numpy.zeros(embedding_length)
            for k, other_answer in enumerate(other_answers):
                # if k != j:
                # other_answer = preprocess_func(other_answer)
                other_answers_emb += retrieve_func(model,
                                                   other_answer,
                                                   embedding_length,
                                                   tfidf_weight)

            answer_emb = retrieve_func(model, answer, embedding_length, tfidf_weight)
            answer_emb = answer_emb - other_answers_emb
            # if not numpy.allclose(zero_embedding, answer_emb) and\
            #         not numpy.allclose(zero_embedding, other_answers_emb):
            #     # print(other_answers_emb)
            #     answer_emb = gram_schmidt(other_answers_emb, answer_emb)

            qa_sim = sim_func(question_emb, answer_emb)
            predictions[i, j] = qa_sim
            # if qa_sim > best_sim:
            #     best_sim = qa_sim
            #     best_answer = j

    #     predictions[i] = best_answer

    # if to_letters:
    #     predictions = numbers_to_letters(predictions)
    # if normalize:
    #     predictions = normalize_preds(predictions)

    return predictions


def predict_answers_by_similarity_grahm(model,
                                        questions,
                                        answers_list,
                                        preprocess_func,
                                        embedding_length,
                                        # to_letters=True,
                                        # normalize=True,
                                        retrieve_func=sum_sentence_embedding,
                                        sim_func=cosine_sim,
                                        tfidf_weight=None):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    # print('\n\n\ngrahm\n',
    #       len(model.vocab),
    #       sorted(model.vocab.keys())[:20],
    #       questions[:20],
    #       answers_list[:20],
    #       preprocess_func,
    #       embedding_length,
    #       retrieve_func,
    #       sim_func,
    #       tfidf_weight)

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        #
        # get and process question
        question = preprocess_func(questions[i])
        question_emb = retrieve_func(model, question, embedding_length, tfidf_weight)

        # if i < 5:
        #     print(retrieve_func)
        #     print(question)
        #     print(question_emb)

        answers = answers_list[i]

        processed_answers = [preprocess_func(a) for a in answers]

        for j, answer in enumerate(answers):

            # answer = preprocess_func(answer)
            # other_answers = [preprocess_func(answers[k]) for k in range(n_answers) if k != j]
            answer = processed_answers[j]
            other_answers = [p for k, p in enumerate(processed_answers) if k != j]

            other_answers = remove_common_tokens(answer, other_answers)

            other_answers_emb = numpy.zeros(embedding_length)
            for k, other_answer in enumerate(other_answers):
                # if k != j:
                # other_answer = preprocess_func(other_answer)
                other_answers_emb += retrieve_func(model,
                                                   other_answer,
                                                   embedding_length,
                                                   tfidf_weight)

            answer_emb = retrieve_func(model, answer, embedding_length, tfidf_weight)
            answer_emb = gram_schmidt(other_answers_emb, answer_emb)
            # answer_emb = answer_emb - other_answers_emb
            # if not numpy.allclose(zero_embedding, answer_emb) and\
            #         not numpy.allclose(zero_embedding, other_answers_emb):
            #     # print(other_answers_emb)
            #     answer_emb = gram_schmidt(other_answers_emb, answer_emb)

            qa_sim = sim_func(question_emb, answer_emb)
            predictions[i, j] = qa_sim
            # if qa_sim > best_sim:
            #     best_sim = qa_sim
            #     best_answer = j

    #     predictions[i] = best_answer

    # if to_letters:
    #     predictions = numbers_to_letters(predictions)
    # if normalize:
    #     predictions = normalize_preds(predictions)

    return predictions


def predict_answers_by_similarity_avg(model,
                                      questions,
                                      answers_list,
                                      preprocess_func,
                                      embedding_length,
                                      # to_letters=True,
                                      # normalize=True,
                                      retrieve_func=sum_sentence_embedding,
                                      sim_func=cosine_sim,
                                      tfidf_weight=None):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        #
        # get and process question
        question = preprocess_func(questions[i])
        question_embs = [retrieve_func(model, [q], embedding_length, tfidf_weight)
                         for q in question]

        # if i < 5:
        #     print(retrieve_func)
        #     print(question)
        #     print(question_emb)

        answers = answers_list[i]

        processed_answers = [preprocess_func(a) for a in answers]

        qa_sim = 0
        for j, answer in enumerate(answers):

            answer = processed_answers[j]

            answer_embs = [retrieve_func(model, a, embedding_length, tfidf_weight)
                           for a in answer]

            for k, (q_emb, a_emb) in enumerate(itertools.combinations((question_embs,
                                                                       answer_embs),
                                                                      2)):
                qa_sim += sim_func(q_emb, a_emb)

            predictions[i, j] = qa_sim / k
            # if qa_sim > best_sim:
            #     best_sim = qa_sim
            #     best_answer = j

    #     predictions[i] = best_answer

    # if to_letters:
    #     predictions = numbers_to_letters(predictions)
    # if normalize:
    #     predictions = normalize_preds(predictions)

    return predictions


def predict_answers_by_similarity_llscore(model,
                                          questions,
                                          answers_list,
                                          preprocess_func,
                                          embedding_length,
                                          # to_letters=True,
                                          # normalize=True,
                                          retrieve_func=sum_sentence_embedding,
                                          sim_func=cosine_sim,
                                          tfidf_weight=None):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        # #
        # # get and process question
        # question = preprocess_func(questions[i])
        # question_emb = retrieve_func(model, question, embedding_length)

        question = questions[i]

        answers = answers_list[i]

        # best_answer = 0  # default to the first one
        # best_sim = -numpy.inf

        # n_answers = len(answers)
        # residuals = []

        # for j, answer in enumerate(answers):
        #     answer = preprocess_func(answer)
        #     residuals.append(retrieve_func(model,
        #                                    get_token_residual(answer, common_part_answers),
        #                                    embedding_length))

        #
        # emb_res(A) -(emb_res(B) + emb_res(C) + emb_res(D))

        qas = []
        for j, answer in enumerate(answers):

            #
            # concatenate q and a
            q_plus_a = question + ' ' + answer

            #
            # and preprocess them
            q_plus_a = preprocess_func(q_plus_a)

            qas.append(q_plus_a)

            # if i < 2:
            #     print(q_plus_a)
            #

        scores = model.score(qas)
        # if i < 2:
        #     print(scores)
        assert len(scores) == len(answers)

        # predictions[i] = numpy.argmax(-scores)
        predictions[i] = scores

    # if to_letters:
    #     predictions = numbers_to_letters(predictions)

    # if normalize:
    #     predictions = normalize_preds(predictions)

    return predictions


def predict_answers_by_most_probable_glove(model,
                                           questions,
                                           answers_list,
                                           preprocess_func,
                                           embedding_length,
                                           # to_letters=True,
                                           # normalize=True,
                                           n_similar=10,
                                           retrieve_func=sum_sentence_embedding,
                                           sim_func=cosine_sim):
    """
        model is a dict-like containing associating words to embeddings

        questions is an iterable of questions

        answers is an iterable of size (n_questions, n_answers)
    """

    n_questions = len(questions)
    assert n_questions == len(answers_list)
    n_answers = len(answers_list[0])

    predictions = numpy.zeros((n_questions, n_answers))

    for i in range(n_questions):
        # #
        # # get and process question
        # question = preprocess_func(questions[i])
        # question_emb = retrieve_func(model, question, embedding_length)

        question = questions[i]
        question = preprocess_func(question)

        most_prob_sentence = model.most_similar_paragraph(question, n_similar)
        most_prob_emb = retrieve_func(model, most_prob_sentence, embedding_length)

        answers = answers_list[i]

        for j, answer in enumerate(answers):

            answer = preprocess_func(answer)
            answer_emb = retrieve_func(answer)

            # predictions[i] = numpy.argmax(-scores)
            predictions[i, j] = sim_func(most_prob_emb, answer_emb)

    return predictions


def data_frame_to_matrix(data_frame, model, retrieve_func, model_type='word2vec'):
    questions_answers, true_answers = get_questions_plus_answers_labelled(data_frame)

    n_observations = len(questions_answers)

    embedding_size = get_model_embedding_size(model, model_type)
    n_features = embedding_size * 2

    data_matrix = numpy.zeros((n_observations, n_features))

    for i, (q, a) in enumerate(questions_answers):
        question_emb = retrieve_func(model, q, embedding_size)
        answer_emb = retrieve_func(model, a, embedding_size)
        data_matrix[i, :] = numpy.hstack((question_emb, answer_emb))

    return data_matrix, numpy.array(true_answers)


def data_frame_to_matrix_q_plus_a(data_frame, model, retrieve_func, model_type='word2vec', ids=None, labels=True):

    true_answers = None
    if labels:
        questions_answers, true_answers = get_questions_plus_answers_labelled(data_frame, ids)
        true_answers = numpy.array(true_answers)

    else:
        questions_answers = get_questions_plus_answers_matrix(data_frame, ids)

    n_observations = len(questions_answers)

    embedding_size = get_model_embedding_size(model, model_type)
    n_features = embedding_size

    data_matrix = numpy.zeros((n_observations, n_features))

    for i, (q, a) in enumerate(questions_answers):
        qa = q + a
        emb = retrieve_func(model, qa, embedding_size)
        data_matrix[i, :] = emb

    return data_matrix, true_answers

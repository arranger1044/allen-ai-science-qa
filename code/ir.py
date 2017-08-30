import numpy
import sklearn

from embeddings import get_model_embedding_size

from time import perf_counter

from embeddings import load_word2vec_model


def corpus_to_vector_space_model(corpus, model, retrieve_func, model_type='word2vec', verbose=False):
    """
    Corpus is a sequence of documents, documents are sequence of preprocessed tokens
    """

    embeddings_size = get_model_embedding_size(model, model_type)
    n_docs = len(corpus)

    model_start_t = perf_counter()
    vector_space = numpy.zeros((n_docs, embeddings_size))

    for i, doc in enumerate(corpus):
        doc_emb = retrieve_func(model, doc, embeddings_size)
        vector_space[i] = doc_emb

    model_end_t = perf_counter()

    if verbose:
        print('Created VSM of size {0} X {1} in {2} secs'.format(n_docs,
                                                                 embeddings_size,
                                                                 model_end_t - model_start_t))

    return vector_space


def query_cosine_similarity(vector_space, query):
    """
    Both vector_space and query are assumed to be normalized
    """

    #
    # check they have the same number of features
    print(query.shape)
    assert vector_space.shape[1] == query.shape[0]

    #
    # cosine similarity
    similarities = numpy.dot(vector_space, query)

    return similarities


def get_n_most_similar_documents_from_corpus(corpus,
                                             query,
                                             n,
                                             model,
                                             retrieve_func,
                                             sim_func=query_cosine_similarity,
                                             model_type='word2vec',
                                             normalize_query=False):
    """
    This version creates a VSM for the corpus but processes only one query, inefficient
    """

    #
    # getting a vsm
    vector_space = corpus_to_vector_space_model(corpus, model, retrieve_func, model_type)
    #
    # normalizing
    vector_space = sklearn.preprocessing.normalize(vector_space)

    #
    # embedding the query
    embeddings_size = get_model_embedding_size(model, model_type)
    query_emb = retrieve_func(model, query, embeddings_size)

    #
    # normalizing query?
    if normalize_query:
        query_emb = sklearn.preprocessing.normalize(query_emb[:, numpy.newaxis], axis=0).ravel()

    #
    # getting similarities
    similarities = sim_func(vector_space, query_emb)

    #
    # getting the top n, descending order
    doc_order = numpy.argsort(-similarities)
    top_n = doc_order[:n]
    print(top_n)

    return [corpus[i] for i in top_n], vector_space[top_n], similarities[top_n]


def n_most_similar_documents(vector_space,
                             queries,
                             model,
                             n,
                             retrieve_func,
                             sim_func=query_cosine_similarity,
                             model_type='word2vec',
                             verbose=False):

    n_queries = len(queries)

    #
    # embedding them
    queries_emb = corpus_to_vector_space_model(queries,
                                               model,
                                               retrieve_func,
                                               model_type,
                                               verbose=verbose)
    #
    # normalizing
    queries_emb = sklearn.preprocessing.normalize(queries_emb)

    #
    # projecting (n_docs, n_features) _dot_ (n_features, n_queries)
    similarities = numpy.dot(vector_space, queries_emb.T)

    #
    # sorting
    doc_orders = numpy.argsort(similarities, axis=0)

    #
    # getting only the top n (rows)
    top_n_docs = doc_orders[:n, :]

    top_similarities = numpy.zeros((n, n_queries))
    for i in range(n_queries):
        top_similarities[:, i] = similarities[top_n_docs[:, i], i]

    return top_n_docs, top_similarities  # , vector_space[top_n_docs]


def rank_answers_by_n_most_similar_documents(corpus,
                                             questions,
                                             answers,
                                             model,
                                             n,
                                             preprocess_func,
                                             retrieve_func,
                                             sim_func,
                                             model_type):
    #
    # preprocessing corpus
    pre_start_t = perf_counter()
    processed_corpus = [preprocess_func(d) for d in corpus]
    pre_end_t = perf_counter()
    print('Processed corpus in {} secs'.format(pre_end_t - pre_start_t))

    #
    # getting a vsm
    vector_space = corpus_to_vector_space_model(processed_corpus,
                                                model,
                                                retrieve_func,
                                                model_type,
                                                verbose=True)
    #
    # normalizing
    vector_space = sklearn.preprocessing.normalize(vector_space)

    n_answers = len(answers[0])
    # predictions = numpy.zeros((len(questions), n_answers))

    queries = []

    for i, q in enumerate(questions):

        #
        # get answers
        for j, ans in enumerate(answers[i]):

            #
            # create a query by concatenating them
            query = q + ' ' + ans
            query = preprocess_func(query)
            queries.append(query)

    assert len(queries) == len(questions) * n_answers

    #
    # now use the vsm to get the scores
    doc_ids, sims = n_most_similar_documents(vector_space=vector_space,
                                             queries=queries,
                                             model=model,
                                             n=n,
                                             retrieve_func=retrieve_func,
                                             sim_func=sim_func,
                                             model_type=model_type,
                                             verbose=True)

    #
    # summing on the rows (the top n scores)
    sim_preds = sims.sum(axis=0)
    assert len(sim_preds) == len(queries)
    # if i < 5:
    #         print(sims_preds, doc_ids)
    predictions = sim_preds.reshape(len(sim_preds) // n_answers,
                                    n_answers)

    return predictions


def rank_answers_by_n_most_similar_documents_multi(corpus,
                                                   questions,
                                                   answers,
                                                   model,
                                                   n,
                                                   preprocess_func,
                                                   retrieve_func,
                                                   sim_func,
                                                   model_type):

    n_questions = len(questions)
    assert n_questions == len(answers)
    n_answers = len(answers[0])

    #
    # preprocessing corpus
    pre_start_t = perf_counter()
    processed_corpus = [preprocess_func(d) for d in corpus]
    pre_end_t = perf_counter()
    print('Processed corpus in {} secs'.format(pre_end_t - pre_start_t))

    #
    # getting a vsm
    vector_space = corpus_to_vector_space_model(processed_corpus,
                                                model,
                                                retrieve_func,
                                                model_type,
                                                verbose=True)
    #
    # normalizing
    vector_space = sklearn.preprocessing.normalize(vector_space)

    n_answers = len(answers[0])
    # predictions = numpy.zeros((len(questions), n_answers))

    queries = []

    for i, q in enumerate(questions):

        query = preprocess_func(q)
        queries.append(query)

        # #
        # # get answers
        # for j, ans in enumerate(answers[i]):

        #     #
        #     # create a query by concatenating them
        #     query = q + ' ' + ans
        #     query = preprocess_func(query)
        #     queries.append(query)

    # assert len(queries) == len(questions) * n_answers

    #
    # now use the vsm to get the scores
    doc_ids, sims = n_most_similar_documents(vector_space=vector_space,
                                             queries=queries,
                                             model=model,
                                             n=n,
                                             retrieve_func=retrieve_func,
                                             sim_func=sim_func,
                                             model_type=model_type,
                                             verbose=True)

    predictions = numpy.zeros((n_questions, n_answers))

    assert doc_ids.shape[1] == len(questions)
    for i, q in enumerate(questions):

        ret_docs = [processed_corpus[k] for k in doc_ids[:, i]]

        ret_docs_embs = [retrieve_func(model, d, model.vector_size) for d in ret_docs]

        for j, ans in enumerate(answers[i]):

            tot_sim_score = 0

            ans = preprocess_func(ans)
            ans_emb = retrieve_func(model, ans, model.vector_size)

            for d_emb in ret_docs_embs:
                tot_sim_score += sim_func(ans_emb, d_emb)

            predictions[i, j] = tot_sim_score
            #
            # summing on the rows (the top n scores)
    # sim_preds = sims.sum(axis=0)
    # assert len(sim_preds) == len(queries)
    # if i < 5:
    #         print(sims_preds, doc_ids)
    # predictions = sim_preds.reshape(len(sim_preds) // n_answers,
    #                                    n_answers)

    return predictions

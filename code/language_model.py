#
# FIXME: this is useless in nltk 3 they removed the code
from nltk.probability import LidstoneProbDist
from nltk.model import NgramModel

from preprocess import tokenize_corpus
from preprocess import lemmatize_tokens
from preprocess import stem_tokens

gamma_smooth = 0.2
lidstone_estimator = lambda fdist, bins: LidstoneProbDist(fdist, gamma_smooth)


from time import perf_counter

# lm = NgramModel(2, brown.words(categories='news'), estimator=estimator)


def lang_model_from_sentences(corpus,
                              n_gram=2,
                              estimator=lidstone_estimator,
                              lemmatize=True,
                              stemming=False):

    #
    # first tokenize the corpus
    token_start_t = perf_counter()
    corpus = tokenize_corpus(corpus)
    token_end_t = perf_counter()
    print('Tokenized corpus in {0} secs'.format(token_end_t - token_start_t))

    #
    # lemmatization and stemming, if necessary
    if lemmatize:
        lemm_start_t = perf_counter()
        corpus = [lemmatize_tokens(d) for d in corpus]
        lemm_end_t = perf_counter()
        print('Lemmatized corpus in {0} secs'.format(lemm_end_t - lemm_start_t))

    if stemming:
        stem_start_t = perf_counter()
        corpus = [stem_tokens(t) for t in corpus]
        stem_end_t = perf_counter()
        print('Stemmed corpus in {0} secs'.format(stem_end_t - stem_start_t))

    #
    # building the n gram model
    model_start_t = perf_counter()
    model = NgramModel(n_gram, corpus, estimator=estimator)
    model_end_t = perf_counter()
    print('{0}-gram model created in {1}'.format(n_gram, model_end_t - model_start_t))
    print(model)

    return model

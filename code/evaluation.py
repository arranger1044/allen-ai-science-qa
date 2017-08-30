import numpy

import pandas

from dataset import load_data_frame
from dataset import get_ids
from dataset import LETTERS
from dataset import LETTER_DICT
from dataset import save_preds_to_csv

from scipy.special import expit

ID_COL = 'id'
ANS_COL = 'correctAnswer'


def compute_accuracy(true_labels, predicted_labels):

    n_observations = true_labels.shape[0]
    assert n_observations == predicted_labels.shape[0]

    return numpy.sum(true_labels == predicted_labels) / n_observations


def normalize_preds(preds, cos_sim=True, eps=0.0):
    """
    an array n_questions X n_answers
    to have its rows normalized in 0-1 (sum to 1)
    """

    # return preds
    if cos_sim:
        # preds = scipy.special.expit(preds)
        preds_min = -1
        preds_max = 1
    else:
        #
        # normalize in [0, 1] with minmax
        preds_min = preds.min() - eps  # [:, numpy.newaxis]
        preds_max = preds.max()  # [:, numpy.newaxis]

    preds = (preds - preds_min) / (preds_max - preds_min)

    #
    # then to have rows to sum to one
    return preds / preds.sum(axis=1)[:, numpy.newaxis]


def min_max_normalize_preds(preds):
    return preds / preds.sum(axis=1)[:, numpy.newaxis]


def norm_sum_preds(preds):
    return preds / preds.sum(axis=1)[:, numpy.newaxis]


def softmax(tensor):

    maxes = numpy.amax(tensor, axis=1, keepdims=True)
    e = numpy.exp(tensor - maxes)
    tensor = e / numpy.sum(e, axis=1, keepdims=True)

    return tensor


def softmax_normalize_preds(preds):

    return softmax(preds)


def hard_preds(preds):
    """
    from an array n_questions X n_answers
    return an array of n_questions integers
    """
    return preds.argmax(axis=1).astype(int)


def random_preds(n_preds, seed=1234):

    rand_gen = numpy.random.RandomState(seed)
    return rand_gen.choice(LETTERS, n_preds)


def invert_preds(preds, pred_ids):
    """
    Invert the preds given an array assumed to be normalized (rows sum to 1)
    """

    preds[pred_ids] = 1 - preds[pred_ids]

    preds[pred_ids] = preds[pred_ids] / preds[pred_ids].sum(axis=1)[:, numpy.newaxis]

    return preds
# def randomly_modify_preds(pred_frame, ids, seed=1234):
#     rand_gen = numpy.random.RandomState(seed)

#     rows_to_modify = pred_frame[ID_COL].isin(ids)
#     print(rows_to_modify)
#     pred_frame.loc[rows_to_modify, ANS_COL] = rand_gen.choice(LETTERS, sum(rows_to_modify))

#     return pred_frame


def modify_preds_by_increment(pred_frame, ids, seed=1234):

    for id in ids:
        old_answer = pred_frame[pred_frame['id'] == id][ANS_COL].values[0]
        new_answer = LETTERS[(LETTER_DICT[old_answer] + 1) % len(LETTERS)]
        pred_frame.loc[pred_frame['id'] == id, ANS_COL] = new_answer
    return pred_frame


def create_submission(preds, output=None, ids=None, test_set_path='../data/test_set.tsv'):

    if ids is None:
        ids = get_ids(load_data_frame(test_set_path))

    assert len(preds) == len(ids)

    pred_frame = pandas.DataFrame({ID_COL: ids, ANS_COL: preds})

    if output:
        save_preds_to_csv(pred_frame, output)

    return pred_frame

import scipy.spatial.distance


def cosine_sim(x, y):
    return 1 - scipy.spatial.distance.cosine(x, y)


def euclidean_sim(x, y):
    return - scipy.spatial.distance.euclidean(x, y)


def manhattan_sim(x, y):
    return - scipy.spatial.distance.cityblock(x, y)

ALMOST_ZERO = 1e-5


def correlation_sim(x, y):
    corr = 1 - scipy.spatial.distance.correlation(x, y)
    if numpy.isnan(corr):
        return ALMOST_ZERO
    return corr


def compute_correlation_of_preds(pred_1, pred_2):
    corr_coef, p_value = scipy.stats.pearsonr(pred_1, pred_2)
    return corr_coef, p_value


def abs_correlation_of_preds(pred_1, pred_2):
    corr_coef, p_value = compute_correlation_of_preds(pred_1, pred_2)
    return abs(corr_coef)


def hard_abs_correlation_of_preds(pred_1, pred_2):
    return abs_correlation_of_preds(hard_preds(pred_1), hard_preds(pred_2))

import numpy

import scipy
import scipy.special
import scipy.stats

import pandas

# from dataset import get_answers
# from dataset import load_predictions
# from dataset import collect_document_paths

# from evaluation import compute_accuracy
# from evaluation import hard_preds
# from evaluation import normalize_preds

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

import itertools
import os
import re

EPS = 1e-10


def compute_accuracy(true_labels, predicted_labels):
    f1_score(true_labels, predicted_labels, average='weighted')


def gen_find(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        # for name in fnmatch.filter(filelist, filepat):
        # for name in [f for f in sorted(filelist) if filepat in f]:
        for name in [f for f in sorted(filelist) if re.search(filepat, f)]:
            yield os.path.join(path, name)


def collect_document_paths(base_dir,
                           doc_ext_pattern):

    return gen_find(doc_ext_pattern, base_dir)


def predictions_to_array(predictions):
    """
    Combining a sequence of predictions (numpy arrays) into a single matrix
    """
    n_predictions = len(predictions)
    assert n_predictions > 1

    #
    # getting the first
    predictions_array = predictions[0]

    #
    # assuming homogeneous inputs
    n_observations = predictions_array.shape[0]

    for other_prediction in predictions[1:]:
        predictions_array = numpy.c_[predictions_array, other_prediction]

    assert predictions_array.shape[0] == n_observations
    assert predictions_array.shape[1] == n_predictions

    return predictions_array


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


def invert_preds(preds, pred_ids):
    """
    Invert the preds given an array assumed to be normalized (rows sum to 1)
    """

    preds[pred_ids] = 1 - preds[pred_ids]

    preds[pred_ids] = preds[pred_ids] / preds[pred_ids].sum(axis=1)[:, numpy.newaxis]

    return preds


def majority_vote_hard_predictions(predictions):

    assert len(predictions) > 1

    n_observations = len(predictions)
    # print(n_observations)
    majority_preds = numpy.zeros(n_observations, dtype=int)
    # print(majority_preds.shape)

    for i in range(n_observations):
        counts = numpy.bincount(predictions[i])
        vote = numpy.argmax(counts)
        majority_preds[i] = vote

    return majority_preds


def averaging_predictions(predictions):
    # return predictions.mean(axis=2)
    n_predictors = predictions.shape[2]
    return predictions.sum(axis=2) / n_predictors
    # sum_preds = predictions.sum(axis=2)
    # return sum_preds / sum_preds.sum(axis=1)[:, numpy.newaxis]


def std_averaging_predictions(predictions):
    std_predictions = (predictions - numpy.mean(predictions, axis=1)[:, numpy.newaxis, :]) / \
        numpy.std(predictions, axis=1, ddof=1)[:, numpy.newaxis, :]
    return averaging_predictions(std_predictions)


def geom_averaging_predictions(predictions):
    return scipy.stats.gmean(predictions, axis=2)


def harm_averaging_predictions(predictions):
    return scipy.stats.hmean(predictions, axis=2)


def median_predictions(predictions):
    return numpy.median(predictions, axis=2)


def circular_averaging_predictions(predictions):
    return scipy.stats.circmean(predictions, axis=2)


def max_predictions(predictions):
    return predictions.max(axis=2)


def min_predictions(predictions):
    return predictions.min(axis=2)


def standard_deviation_predictions(predictions):
    return numpy.std(predictions, axis=2, ddof=1)


def rank_predictions(predictions,
                     method='ordinal'):

    n_items = predictions.shape[0]
    n_predictors = predictions.shape[2]

    ranked_predictions = numpy.array(predictions)
    #
    # first compute ranks
    for i in range(n_items):
        for k in range(n_predictors):
            ranked_predictions[i, :, k] = scipy.stats.rankdata(ranked_predictions[i, :, k],
                                                               method='average')

    return ranked_predictions


def borda_rank_aggregation(predictions,
                           method='average'):

    n_questions = predictions.shape[0]
    n_answers = predictions.shape[1]
    n_predictors = predictions.shape[2]

    scores = numpy.zeros(predictions.shape)

    ordering = numpy.flipud(numpy.arange(n_answers))

    ranked_predictions = rank_predictions(predictions,
                                          method) - 1
    ranked_predictions = ranked_predictions.astype('int')

    for i in range(n_questions):
        for j in range(n_predictors):
            scores[i, ranked_predictions[i, :, j], j] = ordering

    tot_scores = numpy.sum(scores, axis=2)

    return numpy.fliplr(numpy.argsort(tot_scores, axis=1))


def ranked_averaging_predictions(predictions,
                                 method='average'):

    #
    # then averaging
    ranked_predictions = rank_predictions(predictions,
                                          method)
    return averaging_predictions(ranked_predictions)


def weighted_ranked_averaging_predictions(predictions,
                                          method='average'):

    ranked_predictions = rank_predictions(predictions,
                                          method)
    weighted_predictions = predictions * ranked_predictions

    return averaging_predictions(weighted_predictions)


def hard_median_predictions(predictions):
    return hard_preds(median_predictions(predictions))


def hard_weighted_ranked_averaging_predictions(predictions,
                                               method='average'):
    return hard_preds(weighted_ranked_averaging_predictions(predictions, method))


def hard_harm_averaging_predictions(predictions):
    return hard_preds(harm_averaging_predictions(predictions))


def hard_geom_averaging_predictions(predictions):
    return hard_preds(geom_averaging_predictions(predictions))


def hard_ranked_averaging_predictions(predictions):
    return hard_preds(ranked_averaging_predictions(predictions))


def hard_averaging_predictions(predictions):
    return hard_preds(averaging_predictions(predictions))


def hard_circular_averaging_predictions(predictions):
    return hard_preds(circular_averaging_predictions(predictions))


def make_hard_predictions(predictions):
    hard_preds = numpy.zeros((predictions.shape[0], predictions.shape[2]),
                             dtype=int)
    n_predictors = predictions.shape[2]
    for i in range(n_predictors):
        hard_preds[:, i] = predictions[:, :, i].argmax(axis=1)
    return hard_preds


def majority_vote(predictions):
    hard_preds = make_hard_predictions(predictions)
    return majority_vote_hard_predictions(hard_preds)


def weighted_averaging_predictions_unnorm(predictions, weights):
    #
    # normalize weights
    weights = numpy.array(weights)
    weighted_preds = predictions * weights[numpy.newaxis, numpy.newaxis, :]
    return averaging_predictions(weighted_preds)


def weighted_averaging_predictions(predictions, weights):
    #
    # normalize weights
    weights = weights / weights.sum()
    weighted_preds = predictions * weights[numpy.newaxis, numpy.newaxis, :]
    return weighted_preds.mean(axis=2)


def logistic_predictions(predictions, weights):
    #
    # normalize weights
    # weights = weights / weights.sum()
    weighted_preds = predictions * weights[numpy.newaxis, numpy.newaxis, :]
    weighted_preds = weighted_preds.sum(axis=2)
    weighted_preds = scipy.special.expit(weighted_preds)
    return weighted_preds


AGGR_FUNC_DICT = {
    'zsc': std_averaging_predictions,
    'avg': averaging_predictions,
    'maj': majority_vote,
    'rnk': ranked_averaging_predictions,
    'geo': geom_averaging_predictions,
    'hrm': harm_averaging_predictions,
    'wrk': weighted_ranked_averaging_predictions,
    'med': median_predictions,
    'cir': circular_averaging_predictions,
    'max': max_predictions,
    'min': min_predictions,
    'std': standard_deviation_predictions,
    'brd': borda_rank_aggregation,
}


def aggregate_function_factory(func):
    return AGGR_FUNC_DICT[func]


def augment_predictions(preds,
                        aggr_funcs=['avg', 'std', 'rnk', 'geo', 'hrm',
                                    'wrk', 'med', 'min', 'max'],
                        normalize=True,
                        logify=False):

    n_more_funcs = len(aggr_funcs)
    n_predictors = preds.shape[2]

    proc_preds = [preds]
    for aggr_func_type in aggr_funcs:
        #
        # get the func
        aggr_func = aggregate_function_factory(aggr_func_type)

        #
        # aggregate
        aggr_preds = aggr_func(preds)

        if normalize:
            aggr_preds = normalize_preds(aggr_preds, cos_sim=False, eps=EPS)
        #
        # store
        proc_preds.append(aggr_preds)

    #
    # append them all
    aug_preds = numpy.dstack(proc_preds)

    assert aug_preds.shape[2] == n_predictors + n_more_funcs

    if logify:
        n_predictors = aug_preds.shape[2]
        aug_preds = numpy.dstack([aug_preds, numpy.log1p(aug_preds)])
        assert aug_preds.shape[2] == n_predictors * 2

    print('Augmented preds shape:', aug_preds.shape)

    return aug_preds


# def collect_predictions_from_dir_old(pred_dir,
#                                      pred_ext='test.preds'):

#     pred_files = collect_document_paths(pred_dir,
#                                         pred_ext)
#     predictions_list = []
#     for file_path in pred_files:
#         print('Processing file:', file_path)
#         pred_frame = pandas.read_csv(file_path)
#         predictions = get_answers(pred_frame, numeric=True)
#         predictions_list.append(numpy.array(predictions))

#     predictions_matrix = predictions_to_array(predictions_list)

#     majority_preds = majority_vote(predictions_matrix)

#     return majority_preds

# def load_predictions(preds_path, answers=['A', 'B', 'C', 'D']):

#     preds_frame = pandas.read_csv(preds_path)

#     #
#     # order by id
#     preds_frame = preds_frame.sort(ID_COL)

#     answers_col_names = [ANS_CODES[a] for a in answers]

#     #
#     # FIXING for missing values
#     preds_frame = preds_frame.fillna(0.25)

#     return preds_frame[answers_col_names].values


def load_predictions(preds_path):

    return numpy.load(preds_path)


def collect_predictions_from_dir(pred_dir,
                                 pred_ext='test.preds'):

    pred_files = collect_document_paths(pred_dir,
                                        pred_ext)
    predictions_list = []
    for file_path in pred_files:
        print('Processing file:', file_path)
        predictions_list.append(load_predictions(file_path))

    # predictions_matrix = numpy.dstack(tuple(pred for pred in predictions_list))

    # return predictions_matrix

    return collect_predictions(tuple(pred for pred in predictions_list))


def collect_predictions_from_dirs(pred_dirs,
                                  pred_ext='test.preds'):
    predictions_list = []

    for file_path in pred_dirs:
        print('*** Processing dir:', file_path)
        pred_files = collect_document_paths(file_path,
                                            pred_ext)
        for pred_file in pred_files:
            print('Processing file:', pred_file)
            predictions_list.append(load_predictions(pred_file))

    return collect_predictions(tuple(pred for pred in predictions_list))


def predictions_names_from_dirs(pred_dirs,
                                pred_ext='test.preds'):
    predictions_names_list = []

    for file_path in pred_dirs:
        pred_files = collect_document_paths(file_path,
                                            pred_ext)
        for pred_file in pred_files:
            # file_name = pred_file.split('/')[-1]
            file_name = pred_file  # .split('/')[-1]
            predictions_names_list.append(file_name)

    return predictions_names_list


def predictions_paths_from_dirs(pred_dirs,
                                pred_ext='test.preds'):
    predictions_names_list = []

    for file_path in pred_dirs:
        pred_files = collect_document_paths(file_path,
                                            pred_ext)
        for pred_file in pred_files:

            predictions_names_list.append(pred_file)

    return predictions_names_list


def collect_predictions(preds):

    predictions_matrix = numpy.dstack(preds)

    return predictions_matrix


# def backward_greedy_ensemble_search(predictions,
#                                     true_predictions,
#                                     combine_rule,
#                                     metric=compute_accuracy,
#                                     epsilon=0.001):

#     predictions = predictions
#     n_predictors = predictions.shape[2]

#     #
#     #
#     whole_predictions = combine_rule(predictions)
#     whole_score = metric(true_predictions, whole_predictions)

#     print('Considering all predictions: acc {}'.format(whole_score))

#     best_score = whole_score
#     global_best_score = best_score
#     keep_improving = True

#     rem_preds_mask = numpy.ones(n_predictors, dtype=bool)
#     preds_ids = {i for i in range(n_predictors)}

#     global_rem_preds_mask = None

#     while len(preds_ids) > 1 and keep_improving:

#         curr_best_score = 0
#         # curr_best_preds_ids = None
#         # curr_best_preds_mask = None
#         curr_best_pred_id = None

#         #
#         # try to remove each predictor
#         for pred_id in preds_ids:
#             preds_to_keep = numpy.copy(rem_preds_mask)
#             preds_to_keep[pred_id] = False

#             rem_predictions = predictions[:, :, preds_to_keep]

#             aggr_predictions = combine_rule(rem_predictions)
#             #
#             # compute metric
#             rem_score = metric(true_predictions, aggr_predictions)
#             # print(rem_score)

#             if rem_score > curr_best_score:
#                 curr_best_score = rem_score
#                 curr_best_pred_id = pred_id

#         #
#         # any improvement?
#         if curr_best_score - best_score > -epsilon:
#             print('Found an improvement by removing {0}\t [acc:{1}]'.format(curr_best_pred_id,
#                                                                             curr_best_score),
#                   end='          \r')
#             best_score = curr_best_score
#             preds_ids.remove(curr_best_pred_id)
#             rem_preds_mask[curr_best_pred_id] = False

#             if best_score > global_best_score:
#                 global_best_score = best_score
#                 global_rem_preds_mask = numpy.copy(rem_preds_mask)
#         else:
#             print('No more improvements:{}'.format(best_score))
#             keep_improving = False

#     best_preds = combine_rule(predictions[:, :, global_rem_preds_mask])
#     return best_preds, global_best_score, global_rem_preds_mask


# def backward_greedy_ensemble_search_cv(predictions,
#                                        true_predictions,
#                                        combine_rule,
#                                        n_folds=10,
#                                        seed=1337,
#                                        metric=compute_accuracy,
#                                        combine_rule_ens=averaging_predictions,
#                                        voting_scheme=hard_averaging_predictions,
#                                        epsilon=0.001):

#     predictions = predictions
#     n_predictors = predictions.shape[2]
#     n_questions = predictions.shape[0]

#     cv_score = 0
#     kf = KFold(n_questions, n_folds=n_folds, shuffle=True, random_state=seed)

#     models_masks = []
#     for i, (t_ids, v_ids) in enumerate(kf):

#         fold_train_predictions = predictions[t_ids]
#         fold_train_true_predictions = true_predictions[t_ids]
#         train_preds, train_score, train_mask = \
#             backward_greedy_ensemble_search(fold_train_predictions,
#                                             fold_train_true_predictions,
#                                             combine_rule=combine_rule,
#                                             metric=metric)

#         models_masks.append(train_mask)

#         fold_valid_predictions = predictions[v_ids]
#         fold_valid_true_predictions = true_predictions[v_ids]
#         aggr_valid_predictions = combine_rule(fold_valid_predictions[:, :, train_mask])
#         valid_score = metric(fold_valid_true_predictions, aggr_valid_predictions)
#         print('{0}/{1} train: {2} valid: {3}'.format(i + 1, n_folds, train_score, valid_score))
#         print(train_mask)
#         cv_score += valid_score

#     cv_score = cv_score / n_folds

#     print(cv_score)

#     #
#     # intersection
#     intersection_mask = numpy.array(models_masks[0])
#     for i, mask in enumerate(models_masks[1:]):
#         intersection_mask *= mask
#     print('Intersection', sum(intersection_mask))
#     print(intersection_mask)

#     #
#     # counting
#     models_counter = numpy.zeros(n_predictors, dtype=int)
#     for i, mask in enumerate(models_masks):
#         models_counter += mask.astype(int)
#     print(models_counter)

#     min_c = min(models_counter)
#     max_c = max(models_counter)
#     for j in range(min_c, max_c + 1):
#         models = models_counter <= j
#         # ens_preds = numpy.zeros((predictions.shape[0],
#         #                         predictions.shape[1],
#         #                         n_folds))
#         # for i, mask in enumerate(models):
#         model_preds = predictions[:, :, models]
#         aggr_preds = combine_rule(model_preds)
#         # ens_preds[:, :, i] = aggr_preds
#         m_score = metric(true_predictions, aggr_preds)
#         print(j, sum(models), m_score)

#     # aggr_ens_preds = voting_scheme(ens_preds)
#     # aggr_score = metric(true_predictions, aggr_ens_preds)
#     # print('Ens', aggr_score)

#     ens_score = 0
#     ens_preds = numpy.zeros((predictions.shape[0],
#                              predictions.shape[1],
#                              n_folds))
#     for i, mask in enumerate(models_masks):
#         model_preds = predictions[:, :, mask]
#         aggr_preds = combine_rule_ens(model_preds)
#         ens_preds[:, :, i] = aggr_preds
#         m_score = metric(true_predictions, combine_rule(model_preds))
#         print(i, m_score)

#     aggr_ens_preds = voting_scheme(ens_preds)
#     aggr_score = metric(true_predictions, aggr_ens_preds)
#     print('Ens', aggr_score)

#     return cv_score, aggr_score


# def forward_greedy_ensemble_search(predictions,
#                                    true_predictions,
#                                    combine_rule,
#                                    metric=compute_accuracy,
#                                    epsilon=0.001):

#     predictions = predictions
#     n_predictors = predictions.shape[2]

#     #
#     #

#     global_best_score = 0
#     global_rem_preds_mask = None

#     best_score = 0
#     keep_improving = True

#     rem_preds_mask = numpy.zeros(n_predictors, dtype=bool)
#     # whole_ids = {i for i in range(n_predictors)}
#     preds_ids = {i for i in range(n_predictors)}

#     while keep_improving:

#         curr_best_score = 0
#         # curr_best_preds_ids = None
#         # curr_best_preds_mask = None
#         curr_best_pred_id = None

#         #
#         # try to remove each predictor
#         for pred_id in preds_ids:
#             preds_to_keep = numpy.copy(rem_preds_mask)
#             preds_to_keep[pred_id] = True

#             rem_predictions = predictions[:, :, preds_to_keep]

#             aggr_predictions = combine_rule(rem_predictions)
#             #
#             # compute metric
#             rem_score = metric(true_predictions, aggr_predictions)
#             # print(rem_score)

#             if rem_score > curr_best_score:
#                 curr_best_score = rem_score
#                 curr_best_pred_id = pred_id

#         #
#         # any improvement?
#         if curr_best_score - best_score > -epsilon:
#             print('Found an improvement by adding {0} [acc:{1}]'.format(curr_best_pred_id,
#                                                                         curr_best_score))
#             best_score = curr_best_score
#             preds_ids.remove(curr_best_pred_id)
#             rem_preds_mask[curr_best_pred_id] = True

#             if best_score > global_best_score:
#                 global_best_score = best_score
#                 global_rem_preds_mask = numpy.copy(rem_preds_mask)
#         else:
#             print('No more improvements:{}'.format(best_score))
#             keep_improving = False

#     best_preds = combine_rule(predictions[:, :, global_rem_preds_mask])
#     return best_preds, global_best_score, global_rem_preds_mask


# def pairwise_prediction_metric(predictions,
#                                compute_metric,
#                                ascending=True):
#     """
#     Compute a pairwise metric among predictors
#     """

#     n_predictors = predictions.shape[2]

#     pair_scores = []
#     for pred_id_1, pred_id_2 in itertools.combinations(range(n_predictors), 2):
#         #
#         # extract predictors
#         pair_metric = compute_metric(predictions[:, :, pred_id_1],
#                                      predictions[:, :, pred_id_2])

#         pair_scores.append((pair_metric, (pred_id_1, pred_id_2)))

#     #
#     # sorting
#     pair_scores = sorted(pair_scores, key=lambda t: t[0])

#     if not ascending:
#         pair_scores = reversed(pair_scores)

#     return list(pair_scores)


# def forward_metric_pred_selection(predictions,
#                                   compute_metric,
#                                   threshold,
#                                   starting_metric_value=1,
#                                   ascending=True):
#     """

#     """
#     n_predictors = predictions.shape[2]
#     preds_ids = {i for i in range(n_predictors)}

#     #
#     # compute pairwise metrics
#     print('Computing pairwise metrics')
#     pair_scores = pairwise_prediction_metric(predictions,
#                                              compute_metric,
#                                              ascending)
#     # print(pair_scores)

#     #
#     # extracting just first pair, if the score is under the threshold
#     if pair_scores[0][0] < threshold:
#         pred_1, pred_2 = pair_scores[0][1]
#         print('First pair', pred_1, pred_2)

#         selected_preds = numpy.zeros(n_predictors, dtype=bool)
#         selected_preds_ids = set()
#         selected_preds[pred_1] = True
#         selected_preds[pred_2] = True
#         selected_preds_ids.add(pred_1)
#         selected_preds_ids.add(pred_2)

#         preds_ids.remove(pred_1)
#         preds_ids.remove(pred_2)

#         keep_improving = True

#         while keep_improving:
#             #
#             # for each new possible one

#             best_metric_value = starting_metric_value
#             best_pred_id = None

#             keep_improving = False
#             print('iterate')
#             for curr_pred_id in preds_ids:
#                 curr_preds = predictions[:, :, curr_pred_id]
#                 #
#                 # test is for each selected preds
#                 for sel_pred_id in selected_preds_ids:
#                     sel_preds = predictions[:, :, sel_pred_id]

#                     metric_value = compute_metric(curr_preds, sel_preds)
#                     # print(metric_value)
#                     if metric_value > threshold:
#                         print(metric_value)
#                         break
#                 else:
#                     if metric_value < best_metric_value:
#                         best_metric_value = metric_value
#                         best_pred_id = curr_pred_id
#                         keep_improving = True
#             if keep_improving:
#                 #
#                 # adding the best
#                 print('Found {0} with score: {1}'.format(best_pred_id,
#                                                          best_metric_value))
#                 preds_ids.remove(best_pred_id)
#                 selected_preds_ids.add(best_pred_id)
#                 selected_preds[best_pred_id] = True

#     else:
#         print('No pair under the threshold')

#     return selected_preds


# def preds_tensor_to_group(data_tensor, labels):
#     groups = []
#     group_labels = []
#     n_items = data_tensor.shape[0]
#     n_items_per_group = data_tensor.shape[1]

#     for i in range(n_items):
#         #
#         # extracting slices
#         group_i = data_tensor[i, :, :]
#         groups.append(group_i)
#         #
#         # one hot encoding labels
#         if labels is not None:
#             label_i = numpy.zeros(n_items_per_group, dtype=int)
#             label_i[labels[i]] = 1
#             group_labels.append(label_i)

#     assert len(groups) == n_items

#     if labels is not None:
#         assert len(group_labels) == n_items

#     return groups, group_labels

RND_SEED = 6666


class GroupEnsemble():

    def __init__(self, base_model, **base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params

    def fit(self, group_x, group_y, verbose=False):

        assert len(group_x) == len(group_y)

        self.models = []
        #
        # for each group instantiate a model
        for i, (x, y) in enumerate(zip(group_x, group_y)):

            model = self._base_model(**self._base_model_params)

            model.fit(x, y)
            self.models.append(model)

    def predict(self, group_x, aggr_func=numpy.sum):

        assert hasattr(self, 'models')

        n_groups = len(group_x)
        n_items = group_x[0].shape[0]
        n_predictors = len(self.models)
        preds = numpy.zeros((n_groups, n_items, n_predictors))
        for i, x in enumerate(group_x):

            # x_preds = []
            for j, model in enumerate(self.models):
                pred = model.predict(x)
                #
                # normalizing
                # tra_pred = (pred - pred.min() + 1e-10) / (pred.max() - pred.min())
                # pred = tra_pred / tra_pred.sum()
                preds[i, :, j] = pred

            #
            # aggregating

        # preds = normalize_preds(preds, cos_sim=False, eps=1e-10)
        aggr_preds = aggr_func(preds, axis=2)
        # preds.append(x_pred)

        return aggr_preds


class MaskedGroupEnsemble(GroupEnsemble):

    def __init__(self, base_model, feature_masks=None, random_state=None, **base_model_params):

        if random_state is None:
            self._random_state = numpy.random.RandomState(RND_SEED)

        self._feature_masks = feature_masks

        super().__init__(base_model, **base_model_params)

    def fit(self, group_x, group_y, verbose=False):

        assert len(group_x) == len(group_y)

        self.models = []

        n_features = group_x[0].shape[1]
        #
        # if no mask is provided, just build an array of all feature indices
        if self._feature_masks is None:
            self._feature_masks = [numpy.ones(n_features, dtype=bool)]

        for i, (x, y) in enumerate(zip(group_x, group_y)):

            for m, mask in enumerate(self._feature_masks):

                #
                # for each group instantiate a model
                model = self._base_model(**self._base_model_params)

                #
                # masking
                x = x[:, mask]

                #
                # fitting
                model.fit(x, y)

                #
                # storing
                self.models.append((model, m))

    def predict(self, group_x, aggr_func=numpy.sum):

        assert hasattr(self, 'models')

        n_groups = len(group_x)
        n_items = group_x[0].shape[0]
        n_predictors = len(self.models)
        preds = numpy.zeros((n_groups, n_items, n_predictors))
        for i, x in enumerate(group_x):

            # x_preds = []
            for j, (model, mask) in enumerate(self.models):
                #
                # masking
                x = x[:, self._feature_masks[mask]]
                pred = model.predict(x)
                #
                # normalizing
                # tra_pred = (pred - pred.min() + 1e-10) / (pred.max() - pred.min())
                # pred = tra_pred / tra_pred.sum()
                preds[i, :, j] = pred

            #
            # aggregating

        # preds = normalize_preds(preds, cos_sim=False, eps=1e-10)
        aggr_preds = aggr_func(preds, axis=2)
        # preds.append(x_pred)

        return aggr_preds


class RanklibGroupEnsemble():

    def __init__(self, base_model, base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params

    def fit(self, group_x, group_y, group_list, verbose=False):

        assert len(group_x) == len(group_y)
        assert len(group_x) == len(group_list)

        self.models = []
        #
        # for each group instantiate a model
        for i, (x, y, g) in enumerate(zip(group_x, group_y, group_list)):

            #
            # we shall create a file to store the data

            model = self._base_model(**self._base_model_params)

            model.fit(x, y)
            self.models.append(model)

    def predict(self, group_x, aggr_func=numpy.sum):

        assert hasattr(self, 'models')

        n_groups = len(group_x)
        n_items = group_x[0].shape[0]
        n_predictors = len(self.models)
        preds = numpy.zeros((n_groups, n_items, n_predictors))
        for i, x in enumerate(group_x):

            # x_preds = []
            for j, model in enumerate(self.models):
                pred = model.predict(x)
                #
                # normalizing
                # tra_pred = (pred - pred.min() + 1e-10) / (pred.max() - pred.min())
                # pred = tra_pred / tra_pred.sum()
                preds[i, :, j] = pred

            #
            # aggregating

        # preds = normalize_preds(preds, cos_sim=False, eps=1e-10)
        aggr_preds = aggr_func(preds, axis=2)
        # preds.append(x_pred)

        return aggr_preds


class Calibrator():

    def __init__(self, base_model, normalizers=None, **base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params
        self._normalizers = normalizers

    def fit(self, preds, labels):
        n_items = preds.shape[0]
        n_classes = preds.shape[1]
        n_predictors = preds.shape[2]

        self._n_classes = n_classes
        self._n_predictors = n_predictors

        assert len(labels) == n_items

        #
        # transform labels
        label_matrix = numpy.zeros((n_items, n_classes), dtype=int)
        for i in range(n_items):
            label_matrix[i, labels[i]] = 1

        #
        # create models
        self.models = [[None for j in range(n_classes)] for i in range(n_predictors)]

        if self._normalizers is not None:
            norm_preds = [norm(preds) for norm in self._normalizers]

        for j in range(n_classes):
            label_x = label_matrix[:, j]
            for i in range(n_predictors):
                #
                # get data
                train_x = preds[:, j, i]

                if self._normalizers is not None:
                    for norm_train_x in norm_preds:
                        train_x = \
                            numpy.concatenate((train_x.reshape(train_x.shape[0], -1),
                                               norm_train_x.reshape(norm_train_x.shape[0], -1)),
                                              axis=1)
                else:
                    train_x = train_x[:, numpy.newaxis]
                #
                # instantiate model
                model = self._base_model(**self._base_model_params)
                #
                # fit
                # print(train_x.shape, label_x.shape)
                model.fit(train_x, label_x)

                self.models[i][j] = model

    def predict(self, preds):

        assert preds.shape[1] == self._n_classes
        assert preds.shape[2] == self._n_predictors

        if self._normalizers is not None:
            norm_preds = [norm(preds) for norm in self._normalizers]

        calibrated_preds = numpy.zeros(preds.shape)
        for i in range(self._n_predictors):
            for j in range(self._n_classes):

                test_x = preds[:, j, i]
                if self._normalizers is not None:
                    for norm_test_x in norm_preds:
                        test_x = \
                            numpy.concatenate((test_x.reshape(test_x.shape[0], -1),
                                               norm_test_x.reshape(norm_test_x.shape[0], -1)),
                                              axis=1)
                else:
                    test_x = test_x.reshape(-1, 1)

                calibrated_preds[:, j, i] = \
                    self.models[i][j].predict_proba(test_x)[:, 1]

        return calibrated_preds


class OneVsRestClassifier():

    def __init__(self,
                 base_model,
                 feature_selector=None,
                 feature_transformer=None,
                 **base_model_params):

        self._base_model = base_model
        self._base_model_params = base_model_params
        self._feature_selector = feature_selector
        self._feature_transformer = feature_transformer

    def fit(self, preds, labels):

        n_items = preds.shape[0]
        n_classes = preds.shape[1]
        n_predictors = preds.shape[2]

        self._n_classes = n_classes
        self._n_items = n_items
        self._n_predictors = n_predictors

        assert len(labels) == n_items

        #
        # transform labels
        label_matrix = numpy.zeros((n_items, n_classes), dtype=int)
        for i in range(n_items):
            label_matrix[i, labels[i]] = 1

        # print(label_matrix[:20])
        #
        # create models
        self.models = [self._base_model(**self._base_model_params)
                       for i in range(self._n_classes)]

        if self._feature_selector:
            self.feature_selectors = [self._feature_selector['base_selector'](
                **self._feature_selector['base_selector_params'])
                for i in range(self._n_classes)]

        if self._feature_transformer:
            self.feature_transformers = [self._feature_transformer['base_transformer'](
                **self._feature_transformer['base_transformer_params'])
                for i in range(self._n_classes)]
        #
        # fit N models
        for m in range(len(self.models)):

            train_x = preds[:, m, :]
            train_y = label_matrix[:, m]

            #
            # apply feature selection?
            if hasattr(self, 'feature_selectors'):
                # print('feature selection')
                train_x = self.feature_selectors[m].fit_transform(train_x, train_y)

            #
            # transformations?
            if hasattr(self, 'feature_transformers'):
                print('transforming')
                train_x = self.feature_transformers[m].fit_transform(train_x)

            self.models[m].fit(train_x, train_y)

        #
        # storing feature weights
        if self._feature_selector is not None:
            self.feature_importances_ = numpy.zeros(self._n_predictors)
            for m in self.models:
                if hasattr(m, 'coef_'):
                    if m.coef_.ndim == 1:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[p]):
                                self.feature_importances_[p] = abs(m.coef_[p])

                    elif m.coef_.ndim == 2:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[0, p]):
                                self.feature_importances_[p] = abs(m.coef_[0, p])
                            # if abs(self.coef_[p]) < abs(m.coef_[1, p]):
                            #     self.coef_[p] = m.coef_[1, p]
                    else:
                        raise ValueError('More than two classes')

                elif hasattr(m, 'feature_importances_'):
                    if m.feature_importances_.ndim == 1:
                        for p in range(self._n_predictors):
                            if abs(self.feature_importances_[p]) < abs(m.feature_importances_[p]):
                                self.feature_importances_[p] = abs(m.feature_importances_[p])

                    else:
                        raise ValueError('More than two classes')

    def predict(self, preds):

        predicted_probs = numpy.zeros((preds.shape[0], self._n_classes))

        for m in range(len(self.models)):
            test_x = preds[:, m, :]

            if hasattr(self, 'feature_selectors'):
                test_x = self.feature_selectors[m].transform(test_x)

            if hasattr(self, 'feature_transformers'):
                test_x = self.feature_transformers[m].transform(test_x)

            predicted_probs[:, m] = self.models[m].predict_proba(test_x)[:, 1]

        return predicted_probs


class OneVsRestClassifierDiff():

    def __init__(self,
                 base_model,
                 feature_selector=None,
                 feature_transformer=None,
                 **base_model_params):

        self._base_model = base_model
        self._base_model_params = base_model_params
        self._feature_selector = feature_selector
        self._feature_transformer = feature_transformer

    def fit(self, preds, labels):

        n_items = preds.shape[0]
        n_classes = preds.shape[1]
        n_predictors = preds.shape[2]

        self._n_classes = n_classes
        self._n_items = n_items
        self._n_predictors = n_predictors

        assert len(labels) == n_items

        #
        # transform labels
        label_matrix = numpy.zeros((n_items, n_classes), dtype=int)
        for i in range(n_items):
            label_matrix[i, labels[i]] = 1

        # print(label_matrix[:20])
        #
        # create models
        self.models = [self._base_model(**self._base_model_params)
                       for i in range(self._n_classes)]

        if self._feature_selector:
            self.feature_selectors = [self._feature_selector['base_selector'](
                **self._feature_selector['base_selector_params'])
                for i in range(self._n_classes)]

        if self._feature_transformer:
            self.feature_transformers = [self._feature_transformer['base_transformer'](
                **self._feature_transformer['base_transformer_params'])
                for i in range(self._n_classes)]
        #
        # fit N models
        for m in range(len(self.models)):

            train_x = preds[:, m, :]

            for n in range(len(self.models)):
                if m != n:
                    train_x -= preds[:, n, :]

            train_y = label_matrix[:, m]

            #
            # apply feature selection?
            if hasattr(self, 'feature_selectors'):
                # print('feature selection')
                train_x = self.feature_selectors[m].fit_transform(train_x, train_y)

            #
            # transformations?
            if hasattr(self, 'feature_transformers'):
                print('transforming')
                train_x = self.feature_transformers[m].fit_transform(train_x)

            self.models[m].fit(train_x, train_y)

        #
        # storing feature weights
        if self._feature_selector is not None:
            self.feature_importances_ = numpy.zeros(self._n_predictors)
            for m in self.models:
                if hasattr(m, 'coef_'):
                    if m.coef_.ndim == 1:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[p]):
                                self.feature_importances_[p] = abs(m.coef_[p])

                    elif m.coef_.ndim == 2:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[0, p]):
                                self.feature_importances_[p] = abs(m.coef_[0, p])
                            # if abs(self.coef_[p]) < abs(m.coef_[1, p]):
                            #     self.coef_[p] = m.coef_[1, p]
                    else:
                        raise ValueError('More than two classes')

                elif hasattr(m, 'feature_importances_'):
                    if m.feature_importances_.ndim == 1:
                        for p in range(self._n_predictors):
                            if abs(self.feature_importances_[p]) < abs(m.feature_importances_[p]):
                                self.feature_importances_[p] = abs(m.feature_importances_[p])

                    else:
                        raise ValueError('More than two classes')

    def predict(self, preds):

        predicted_probs = numpy.zeros((preds.shape[0], self._n_classes))

        for m in range(len(self.models)):
            test_x = preds[:, m, :]

            for n in range(len(self.models)):
                if m != n:
                    test_x -= preds[:, n, :]

            if hasattr(self, 'feature_selectors'):
                test_x = self.feature_selectors[m].transform(test_x)

            if hasattr(self, 'feature_transformers'):
                test_x = self.feature_transformers[m].transform(test_x)

            predicted_probs[:, m] = self.models[m].predict_proba(test_x)[:, 1]

        return predicted_probs


def preds_difference(preds):
    return preds[:, 0, :] - preds[:, 1, :]


def preds_ratio(preds):
    return preds[:, 0, :] / (preds[:, 1, :] + 1e-10)


def aug_preds_difference(preds):
    preds_diffs = preds_difference(preds)
    return numpy.hstack((preds.reshape(preds.shape[0], -1), preds_diffs))


class OneVsOneClassifier():

    def __init__(self, base_model, feature_selector=None, aggr_func=None, **base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params
        self._feature_selector = feature_selector
        self._aggr_func = aggr_func

    def fit(self, preds, labels):

        n_items = preds.shape[0]
        n_classes = preds.shape[1]
        n_predictors = preds.shape[2]

        self._n_classes = n_classes
        self._n_items = n_items
        self._n_predictors = n_predictors

        assert len(labels) == n_items

        #
        # transform labels
        label_matrix = numpy.zeros((n_items, n_classes), dtype=int)
        for i in range(n_items):
            label_matrix[i, labels[i]] = 1

        self.label_indices = label_matrix.astype(bool)

        #
        # taking all pairwise interactions (6)
        self.model_indexes = list(itertools.combinations(range(self._n_classes), 2))
        self.n_models = len(self.model_indexes)
        # print(label_matrix[:20])
        # range(len(self.models))
        # create models
        self.models = [self._base_model(**self._base_model_params)
                       for i in range(self.n_models)]

        if self._feature_selector:
            self.feature_selectors = [self._feature_selector['base_selector'](
                **self._feature_selector['base_selector_params'])
                for i in range(self.n_models)]
        #
        # fit N models
        for c, (m_1, m_2) in enumerate(self.model_indexes):

            joint_indices = self.label_indices[:, m_1] + self.label_indices[:, m_2]

            train_x = preds[joint_indices, :, :]
            train_x = train_x[:, [m_1, m_2], :]

            if self._aggr_func is None:
                train_x = train_x.reshape(train_x.shape[0], -1)
            else:
                train_x = self._aggr_func(train_x)

            #
            # m_1 = 0, m_2 = 1
            train_y = label_matrix[joint_indices, :]
            train_y = train_y[:, m_2]
            # print(train_y.shape)

            #
            # apply feature selection?
            if hasattr(self, 'feature_selectors'):
                # print('feature selection')
                raise ValueError('Not implemented yet')
                # train_x = self.feature_selectors[m].fit_transform(train_x, train_y)

            self.models[c].fit(train_x, train_y)

        if self._feature_selector is not None:
            self.feature_importances_ = numpy.zeros(self._n_predictors)
            for m in self.models:
                if hasattr(m, 'coef_'):
                    if m.coef_.ndim == 1:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[p]):
                                self.feature_importances_[p] = abs(m.coef_[p])

                    elif m.coef_.ndim == 2:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[0, p]):
                                self.feature_importances_[p] = abs(m.coef_[0, p])
                            # if abs(self.coef_[p]) < abs(m.coef_[1, p]):
                            #     self.coef_[p] = m.coef_[1, p]
                    else:
                        raise ValueError('More than two classes')

                elif hasattr(m, 'feature_importances_'):
                    if m.feature_importances_.ndim == 1:
                        for p in range(self._n_predictors):
                            if abs(self.feature_importances_[p]) < abs(m.feature_importances_[p]):
                                self.feature_importances_[p] = abs(m.feature_importances_[p])

                    else:
                        raise ValueError('More than two classes')

    def predict(self, preds):

        predicted_probs = numpy.zeros((preds.shape[0], self._n_classes))

        for c, (m_1, m_2) in enumerate(self.model_indexes):

            test_x = preds[:, [m_1, m_2], :]
            # test_x = preds
            if self._aggr_func is None:
                test_x = test_x.reshape(test_x.shape[0], -1)
            else:
                test_x = self._aggr_func(test_x)

            if hasattr(self, 'feature_selectors'):
                raise ValueError('Not implemented yet')
                # test_x = self.feature_selectors[m].transform(test_x)

            # # predicted_probs[:, m] = self.models[m].predict_proba(test_x)[:, 1]
            # test_preds = self.models[c].predict(test_x)
            # for i in range(preds.shape[0]):
            #     if test_preds[i] == 0:
            #         predicted_probs[i, m_1] += 1
            #     elif test_preds[i] == 1:
            #         predicted_probs[i, m_2] += 1

            test_preds_0 = self.models[c].predict_proba(test_x)[:, 0]
            test_preds_1 = self.models[c].predict_proba(test_x)[:, 1]
            predicted_probs[:, m_1] += test_preds_0
            predicted_probs[:, m_2] += test_preds_1
            # for i in range(preds.shape[0]):
            #     if test_preds[i] == 0:
            #         predicted_probs[i, m_1] += 1
            #     elif test_preds[i] == 1:
            #         predicted_probs[i, m_2] += 1

        #
        # normalizing into probs again
        predicted_probs = predicted_probs / predicted_probs.sum(axis=1)[:, None]

        return predicted_probs


class OneVsOneClassifierDiff(OneVsOneClassifier):

    def __init__(self, base_model, feature_selector=None, **base_model_params):
        super().__init__(base_model=base_model,
                         feature_selector=feature_selector,
                         aggr_func=preds_difference,
                         **base_model_params)


class OneVsOneClassifier_hop():

    def __init__(self, base_model, feature_selector=None, aggr_func=None, **base_model_params):
        self._base_model = base_model
        self._base_model_params = base_model_params
        self._feature_selector = feature_selector
        self._aggr_func = aggr_func

    def fit(self, preds, labels):

        n_items = preds.shape[0]
        n_classes = preds.shape[1]
        n_predictors = preds.shape[2]

        self._n_classes = n_classes
        self._n_items = n_items
        self._n_predictors = n_predictors

        assert len(labels) == n_items

        #
        # transform labels
        label_matrix = numpy.zeros((n_items, n_classes), dtype=int)
        for i in range(n_items):
            label_matrix[i, labels[i]] = 1

        self.label_indices = label_matrix.astype(bool)

        #
        # taking all pairwise interactions (6)
        self.model_indexes = list(itertools.combinations(range(self._n_classes), 2))
        self.n_models = len(self.model_indexes)
        # print(label_matrix[:20])
        # range(len(self.models))
        # create models
        self.models = [self._base_model(**self._base_model_params)
                       for i in range(self.n_models)]

        if self._feature_selector:
            self.feature_selectors = [self._feature_selector['base_selector'](
                **self._feature_selector['base_selector_params'])
                for i in range(self.n_models)]
        #
        # fit N models
        for c, (m_1, m_2) in enumerate(self.model_indexes):

            joint_indices = self.label_indices[:, m_1] + self.label_indices[:, m_2]

            # train_x_m_1 = preds[self.label_indices[:, m_1], m_1, :]
            # train_x_m_2 = preds[self.label_indices[:, m_2], m_2, :]

            # train_x = numpy.vstack((train_x_m_1, train_x_m_2))

            train_x = preds[joint_indices, :, :]
            # train_x = train_x[:, [m_1, m_2], :]

            if self._aggr_func is None:
                train_x = train_x.reshape(train_x.shape[0], -1)
            else:
                train_x = self._aggr_func(train_x)

            #
            # m_1 = 0, m_2 = 1
            # train_y_m_1 = label_matrix[self.label_indices[:, m_1], m_2][:, None]
            # train_y_m_2 = label_matrix[self.label_indices[:, m_2], m_2][:, None]
            # train_y = numpy.vstack((train_y_m_1, train_y_m_2))
            train_y = label_matrix[joint_indices, :]
            train_y = train_y[:, m_2]
            # print(train_y.shape)

            #
            # apply feature selection?
            if hasattr(self, 'feature_selectors'):
                # print('feature selection')
                raise ValueError('Not implemented yet')
                # train_x = self.feature_selectors[m].fit_transform(train_x, train_y)

            self.models[c].fit(train_x, train_y)

        if self._feature_selector is not None:
            self.feature_importances_ = numpy.zeros(self._n_predictors)
            for m in self.models:
                if hasattr(m, 'coef_'):
                    if m.coef_.ndim == 1:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[p]):
                                self.feature_importances_[p] = abs(m.coef_[p])

                    elif m.coef_.ndim == 2:
                        for p in range(self._n_predictors):
                            if self.feature_importances_[p] < abs(m.coef_[0, p]):
                                self.feature_importances_[p] = abs(m.coef_[0, p])
                            # if abs(self.coef_[p]) < abs(m.coef_[1, p]):
                            #     self.coef_[p] = m.coef_[1, p]
                    else:
                        raise ValueError('More than two classes')

                elif hasattr(m, 'feature_importances_'):
                    if m.feature_importances_.ndim == 1:
                        for p in range(self._n_predictors):
                            if abs(self.feature_importances_[p]) < abs(m.feature_importances_[p]):
                                self.feature_importances_[p] = abs(m.feature_importances_[p])

                    else:
                        raise ValueError('More than two classes')

    def predict(self, preds):

        predicted_probs = numpy.zeros((preds.shape[0], self._n_classes))

        for c, (m_1, m_2) in enumerate(self.model_indexes):

            # for k in range(self._n_classes):

            #     test_x = preds[:, k, :]
            # test_x = preds[:, [m_1, m_2], :]
            test_x = preds
            if self._aggr_func is None:
                test_x = test_x.reshape(test_x.shape[0], -1)
            else:
                test_x = self._aggr_func(test_x)

            if hasattr(self, 'feature_selectors'):
                raise ValueError('Not implemented yet')
            # test_x = self.feature_selectors[m].transform(test_x)

            # # predicted_probs[:, m] = self.models[m].predict_proba(test_x)[:, 1]
            # test_preds = self.models[c].predict(test_x)
            # for i in range(preds.shape[0]):
            #     if test_preds[i] == 0:
            #         predicted_probs[i, m_1] += 1
            #     elif test_preds[i] == 1:
            #         predicted_probs[i, m_2] += 1

            test_preds_0 = self.models[c].predict_proba(test_x)[:, 0]
            test_preds_1 = self.models[c].predict_proba(test_x)[:, 1]
            predicted_probs[:, m_1] += test_preds_0
            predicted_probs[:, m_2] += test_preds_1
            # for i in range(preds.shape[0]):
            #     if test_preds[i] == 0:
            #         predicted_probs[i, m_1] += 1
            #     elif test_preds[i] == 1:
            #         predicted_probs[i, m_2] += 1

        #
        # normalizing into probs again
        predicted_probs = predicted_probs / predicted_probs.sum(axis=1)[:, None]

        return predicted_probs

from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model


def meta_classifier_cv_score(train_preds,
                             train_labels,
                             seeds=[1337, 6666, 7777, 5555],
                             n_folds=10,
                             meta_classifier=OneVsRestClassifier,
                             model_dict={
                                 "base_model": linear_model.LogisticRegression,
                                 "base_model_params": {
                                     "fit_intercept": True,
                                     "class_weight": "balanced",
                                     "penalty": "l2",
                                     "C": 10.0,
                                     "max_iter": 200}
                             }):

    seed_valid_acc_list = []

    for r, seed in enumerate(seeds):

        kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

        # rand_gen = random.Random(seed)
        numpy_rand_gen = numpy.random.RandomState(seed)

        cv_valid_accs = []
        cv_train_accs = []

        for k, (train_ids, test_ids) in enumerate(kf):

            # print('Fold', k)

            train_x = train_preds[train_ids]
            train_y = train_labels[train_ids]
            test_x = train_preds[test_ids]
            test_y = train_labels[test_ids]

            model = meta_classifier(model_dict['base_model'],
                                    feature_selector=None,  # base_feature_sel_dict,
                                    **model_dict['base_model_params'])

            #
            # fitting
            # print('Fitting')
            model.fit(train_x, train_y)

            #
            # predicting
            # print('Predicting on test')
            test_pred_probs = model.predict(test_x)
            hard_test_preds = hard_preds(test_pred_probs)

            test_acc = compute_accuracy(test_y, hard_test_preds)
            cv_valid_accs.append(test_acc)

        avg_valid_acc = sum(cv_valid_accs) / n_folds
        # print('\tAVG on TEST', avg_valid_acc, end='   \r')
        seed_valid_acc_list.append(avg_valid_acc)

    avg_seed_valid_acc = sum(seed_valid_acc_list) / len(seeds)
    return avg_seed_valid_acc


def backward_greedy_preds_search_cv(predictions,
                                    true_predictions,
                                    seeds=[1337, 6666, 7777, 5555],
                                    metric=compute_accuracy,
                                    n_folds=5,
                                    meta_classifier=OneVsRestClassifier,
                                    model_dict={
                                        "base_model": linear_model.LogisticRegression,
                                        "base_model_params": {
                                            "fit_intercept": True,
                                            "class_weight": "balanced",
                                            "penalty": "l2",
                                            "C": 10.0,
                                            "max_iter": 200}
                                    },
                                    epsilon=0.001):

    predictions = predictions
    n_predictors = predictions.shape[2]

    #
    #
    whole_score = meta_classifier_cv_score(predictions,
                                           true_predictions,
                                           seeds=seeds,
                                           n_folds=n_folds,
                                           meta_classifier=meta_classifier,
                                           model_dict=model_dict)

    print('Considering all predictions: acc {}'.format(whole_score))

    best_score = whole_score
    global_best_score = best_score
    keep_improving = True

    rem_preds_mask = numpy.ones(n_predictors, dtype=bool)
    preds_ids = {i for i in range(n_predictors)}

    global_rem_preds_mask = None

    while len(preds_ids) > 1 and keep_improving:

        curr_best_score = 0
        # curr_best_preds_ids = None
        # curr_best_preds_mask = None
        curr_best_pred_id = None

        #
        # try to remove each predictor
        for i, pred_id in enumerate(preds_ids):
            preds_to_keep = numpy.copy(rem_preds_mask)
            preds_to_keep[pred_id] = False

            rem_predictions = predictions[:, :, preds_to_keep]

            rem_score = meta_classifier_cv_score(rem_predictions,
                                                 true_predictions,
                                                 seeds=seeds,
                                                 n_folds=n_folds,
                                                 meta_classifier=meta_classifier,
                                                 model_dict=model_dict)
            #
            # compute metric
            # rem_score = metric(true_predictions, aggr_predictions)
            # print(rem_score)

            print('{0}/{1}:{2}'.format(i + 1, len(preds_ids), rem_score), end='       \r')
            if rem_score > curr_best_score:
                curr_best_score = rem_score
                curr_best_pred_id = pred_id

        #
        # any improvement?
        if curr_best_score - best_score > -epsilon:
            print('Found an improvement by removing {0}\t [acc:{1}]'.format(curr_best_pred_id,
                                                                            curr_best_score),
                  # end='          \r'
                  )
            best_score = curr_best_score
            preds_ids.remove(curr_best_pred_id)
            rem_preds_mask[curr_best_pred_id] = False

            if best_score > global_best_score:
                global_best_score = best_score
                global_rem_preds_mask = numpy.copy(rem_preds_mask)
        else:
            print('No more improvements:{}'.format(best_score))
            keep_improving = False

    # best_preds = combine_rule(predictions[:, :, global_rem_preds_mask])
    return global_best_score, global_rem_preds_mask


def feature_importances_meta_clf(train_preds,
                                 train_labels,
                                 meta_classifier=OneVsRestClassifier,
                                 model_dict={
                                     "base_model": linear_model.LogisticRegression,
                                     "base_model_params": {
                                         "fit_intercept": True,
                                         "class_weight": "balanced",
                                         "penalty": "l2",
                                         "C": 10.0,
                                         "max_iter": 200}
                                 }):
    model = meta_classifier(model_dict['base_model'],
                            feature_selector={},  # base_feature_sel_dict,
                            **model_dict['base_model_params'])

    #
    # fitting
    # print('Fitting')
    model.fit(train_preds, train_labels)

    return model.feature_importances_


def feature_importances_meta_clf_cv(train_preds,
                                    train_labels,
                                    seeds=[1337, 6666, 7777, 5555],
                                    n_folds=5,
                                    meta_classifier=OneVsRestClassifier,
                                    model_dict={
                                        "base_model": linear_model.LogisticRegression,
                                        "base_model_params": {
                                            "fit_intercept": True,
                                            "class_weight": "balanced",
                                            "penalty": "l2",
                                            "C": 10.0,
                                            "max_iter": 200}
                                    }):

    n_predictors = train_preds.shape[2]

    feature_importances = numpy.zeros(n_predictors)

    for r, seed in enumerate(seeds):

        kf = StratifiedKFold(train_labels, n_folds=n_folds, shuffle=True, random_state=seed)

        # rand_gen = random.Random(seed)
        numpy_rand_gen = numpy.random.RandomState(seed)

        for k, (train_ids, test_ids) in enumerate(kf):

            # print('Fold', k)

            train_x = train_preds[train_ids]
            train_y = train_labels[train_ids]

            model_feature_importances = feature_importances_meta_clf(train_x,
                                                                     train_y,
                                                                     meta_classifier,
                                                                     model_dict)

            #
            # always getting the max importance, being conservative
            for p in range(n_predictors):
                if abs(feature_importances[p]) < abs(model_feature_importances[p]):
                    feature_importances[p] = abs(model_feature_importances[p])

    #
    # min, max scaling
    scaled_feature_importances = ((feature_importances - numpy.min(feature_importances)) /
                                  (numpy.max(feature_importances) - numpy.min(feature_importances)))

    return scaled_feature_importances

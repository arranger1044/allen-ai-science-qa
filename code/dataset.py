
import numpy
import pandas

import os
import fnmatch
import re

from time import perf_counter

ID_COL = 'id'
QUESTION_COL = 'question'
CORR_ANS_COL = 'correctAnswer'
ANS_A_COL = 'answerA'
ANS_B_COL = 'answerB'
ANS_C_COL = 'answerC'
ANS_D_COL = 'answerD'

ANS_IDS = {ANS_A_COL: 0,
           ANS_B_COL: 1,
           ANS_C_COL: 2,
           ANS_D_COL: 4}

ANS_CODES = {'A': ANS_A_COL,
             'B': ANS_B_COL,
             'C': ANS_C_COL,
             'D': ANS_D_COL}

LETTERS = ['A', 'B', 'C', 'D']
LETTER_DICT = {'A': 0,
               'B': 1,
               'C': 2,
               'D': 3}


def letters_to_numbers(preds,
                       translation_dict=LETTER_DICT):

    return numpy.array([translation_dict[l] for l in preds])


def numbers_to_letters(preds,
                       translation_dict=LETTERS):
    return numpy.array([translation_dict[n] for n in preds])


def load_data_frame(data_path, sep='\t', verbose=True):
    """
    Loading a csv-like file with pandas
    """
    load_start_t = perf_counter()
    data_frame = pandas.read_csv(data_path, sep=sep)
    load_end_t = perf_counter()

    if verbose:
        print('Loaded datasets with {0} X {1} in {2} secs\n'.format(data_frame.shape[0],
                                                                    data_frame.shape[1],
                                                                    load_end_t - load_start_t))

    return data_frame


def save_predictions_numpy(preds, output):
    numpy.savetxt(output, preds, fmt='%.10e', delimiter=',')


def load_predictions_numpy(preds_path):
    return numpy.loadtxt(preds_path, delimiter=',')


def save_preds_to_csv(preds_frame, output):
    preds_frame.to_csv(output, header=True, index=False)


def save_predictions(preds,
                     output,
                     ids=None,
                     answers=['A', 'B', 'C', 'D'],
                     # test_set_path='../data/validation_set.tsv',
                     test_set_path='../data/test_set.tsv'):

    answers_col_names = [ANS_CODES[a] for a in answers]

    preds_frame = pandas.DataFrame(columns=[ID_COL] + answers_col_names)

    if ids is None:
        ids = get_ids(load_data_frame(test_set_path))

    # preds_frame.to_csv(output, header=True, index=False)
    preds_frame[ID_COL] = ids
    preds_frame[answers_col_names] = preds

    save_preds_to_csv(preds_frame, output)

    return preds_frame


def load_predictions(preds_path, answers=['A', 'B', 'C', 'D']):

    preds_frame = pandas.read_csv(preds_path)

    #
    # order by id
    preds_frame = preds_frame.sort(ID_COL)

    answers_col_names = [ANS_CODES[a] for a in answers]

    #
    # FIXING for missing values
    preds_frame = preds_frame.fillna(0.25)

    return preds_frame[answers_col_names].values


def get_column_values_from_dataframe(data_frame, col, as_list=True):
    """
    Extracting the values in a column, optionally returning a list of
    those values
    """
    column_values = data_frame[col].values

    if as_list:
        column_values = list(column_values)

    return column_values


def get_ids(data_frame):
    """
    Extracting only question ids
    """
    ids = get_column_values_from_dataframe(data_frame, ID_COL)

    assert len(ids) == data_frame.shape[0]

    return ids


def get_answers(data_frame, numeric=False, as_list=False):
    """
    Extract correct answers as a numpy array
    Optionally as integers
    """
    answers = get_column_values_from_dataframe(data_frame, CORR_ANS_COL, as_list)

    if numeric:
        answers = letters_to_numbers(answers)
    return answers


def get_questions(data_frame, ids=False):
    """
    Extracting only questions from the datasets
    As a numpy array of strings
    """
    questions = get_column_values_from_dataframe(data_frame, QUESTION_COL)

    assert len(questions) == data_frame.shape[0]

    if ids:
        ids = get_column_values_from_dataframe(data_frame, ID_COL)
        assert len(ids) == data_frame.shape[0]
        questions = [(id, q) for id, q in zip(ids, questions)]

    return questions


def get_data_matrix(data_frame,
                    columns=[QUESTION_COL,
                             ANS_A_COL,
                             ANS_B_COL,
                             ANS_C_COL,
                             ANS_D_COL],
                    as_list=False):

    matrix = get_column_values_from_dataframe(data_frame, columns, as_list)

    return matrix


def get_possible_answers(data_frame, answers=['A', 'B', 'C', 'D']):

    answers_col_names = [ANS_CODES[a] for a in answers]

    return get_column_values_from_dataframe(data_frame, answers_col_names)


def get_questions_plus_answers(data_frame, answers=['A', 'B', 'C', 'D']):
    """
    Extracting questions + answers specified in a list
    """
    questions = get_questions(data_frame)

    all_answers = get_possible_answers(data_frame, answers)

    qas = []

    for i, q in enumerate(questions):

        assert len(all_answers[i]) == len(answers)

        qas_i = []
        for a in all_answers[i]:

            qas_i.append(q + ' ' + a)

        qas.append(qas_i)

    return qas


def get_questions_plus_answers_labelled(data_frame, ids=None):
    """
    Extracting questions + answers plus a label to say if they are correct or no
    """
    questions = get_questions(data_frame)

    all_answers = get_possible_answers(data_frame, answers=['A', 'B', 'C', 'D'])

    answers = get_answers(data_frame, numeric=True)

    if ids is not None:
        questions = numpy.array(questions)[ids]
        all_answers = numpy.array(all_answers)[ids]
        answers = numpy.array(answers)[ids]

    qas = []
    true_answers = []

    for i, q in enumerate(questions):

        qas_i = []
        for j, a in enumerate(all_answers[i]):

            correct_answer = 0 if answers[i] != j else 1

            qas_i.append((q, a))
            true_answers.append(correct_answer)

        qas.extend(qas_i)

    assert len(qas) == len(true_answers)

    return qas, true_answers


def get_questions_plus_answers_matrix(data_frame, ids=None):
    """
    Extracting questions + answers plus a label to say if they are correct or no
    """
    questions = get_questions(data_frame)

    all_answers = get_possible_answers(data_frame, answers=['A', 'B', 'C', 'D'])

    if ids is not None:
        questions = numpy.array(questions)[ids]
        all_answers = numpy.array(all_answers)[ids]

    qas = []

    for i, q in enumerate(questions):

        qas_i = []
        for j, a in enumerate(all_answers[i]):

            qas_i.append((q, a))

        qas.extend(qas_i)

    return qas


def get_questions_plus_correct_answer(data_frame):
    """
    """

    qas = get_questions_plus_answers(data_frame)

    answers = get_answers(data_frame, numeric=True)

    correct_qas = [qa[answer_id] for qa, answer_id in zip(qas, answers)]

    return correct_qas


def filter_by_ids(data_frame, ids):
    return data_frame.loc[data_frame[ID_COL].isin(ids)]


def gen_find(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        # for name in fnmatch.filter(filelist, filepat):
        # for name in [f for f in sorted(filelist) if filepat in f]:
        for name in [f for f in sorted(filelist) if re.search(filepat, f)]:
            yield os.path.join(path, name)


def collect_document_paths(base_dir,
                           doc_ext_pattern):

    return gen_find(doc_ext_pattern, base_dir)


def load_ck12_concepts_book(file_path='../data/ck12.org/concepts/concepts.txt',
                            lines=False,
                            remove_newlines=True):

    concepts = None
    with open(file_path, 'r') as f:
        concepts = f.read()

    if remove_newlines:
        concepts = concepts.replace('\n', ' ')

    return concepts


def load_ck12_corpus(base_path,
                     ext='.txt'):
    """
    Each document is a paragraph in the text
    But first of all load all chapters
    """
    chapter_paths = [p for p in collect_document_paths(ext, base_path)]
    chapters = []
    for path in chapter_paths:
        with open(path) as chapter_file:
            content = chapter_file.readlines()
            chapters.append()


def load_tensors_preds(pred_files):
    """
    Loading 3d tensors from numpy binaries and stack them on the third axis
    """
    pred_tensors = [numpy.load(preds) for preds in pred_files]
    return numpy.dstack(pred_tensors)


def load_tensors_preds_from_files(pred_dirs, file_exts):
    pred_paths = [collect_document_paths(pred_dir, file_exts) for pred_dir in pred_dirs]
    pred_paths = [path for paths in pred_paths for path in paths]
    return load_tensors_preds(pred_paths)


def load_feature_map(feature_file,
                     # feature_ids_cols=['except', 'noneabove',
                     #                   'allabove', 'notq',
                     #                   'nota', 'what',
                     #                   'how', 'which',
                     #                   'when', 'why'],
                     id_cols=['qid', 'aid'],
                     n_answers=4):
    feature_frame = pandas.read_csv(feature_file)
    feature_frame = feature_frame.sort(id_cols)
    feature_ids_cols = [col for col in feature_frame.columns if col not in id_cols]
    feature_matrix = feature_frame[feature_ids_cols].values
    n_questions = feature_matrix.shape[0] // n_answers
    n_predictors = feature_matrix.shape[1]
    feature_tensor = numpy.zeros((n_questions, n_answers, n_predictors))
    for i in range(n_questions):
        for j in range(n_answers):
            feature_tensor[i, j, :] = feature_matrix[i * n_answers + j, :]

    # feature_tensor = feature_tensor / (feature_tensor.sum(axis=1, keepdims=True) + 1e-10)
    return feature_tensor


def load_feature_maps_from_files(maps_dirs, file_exts):
    maps_paths = [collect_document_paths(dir, file_exts) for dir in maps_dirs]
    maps_paths = [path for paths in maps_paths for path in paths]
    feature_maps = [load_feature_map(path) for path in maps_paths]
    return feature_maps


def stack_feature_tensors(preds_tensor, feature_maps):
    for f_map in feature_maps:
        preds_tensor = numpy.dstack((preds_tensor, f_map))

    return preds_tensor

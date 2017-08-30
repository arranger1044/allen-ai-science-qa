import numpy
from numpy.testing import assert_array_almost_equal

import pandas

import os

from dataset import collect_document_paths


def same_predictors(pred_file_1, pred_file_2,
                    cols=['answerA', 'answerB', 'answerC', 'answerD'],
                    rows=8130):
    #
    # load frames
    pred_frame_1 = pandas.read_csv(pred_file_1)
    pred_frame_2 = pandas.read_csv(pred_file_2)

    #
    # extract pred values
    pred_values_1 = pred_frame_1[cols].values[:rows]
    pred_values_2 = pred_frame_2[cols].values[:rows]

    #
    # compare them
    try:
        assert_array_almost_equal(pred_values_1, pred_values_2)
        return True
    except:
        return False


def compare_predictors_dirs(pred_dir_1, pred_dir_2, ext_1, ext_2, pattern_1, pattern_2):
    #
    # collecting files first
    pred_files_1 = collect_document_paths(pred_dir_1, ext_1)
    pred_files_2 = collect_document_paths(pred_dir_2, ext_2)

    #
    # then removing their paths
    pred_path_dict_1 = {os.path.basename(p).replace(pattern_1, ''): p for p in pred_files_1}
    pred_path_dict_2 = {os.path.basename(p).replace(pattern_2, ''): p for p in pred_files_2}

    #
    # cheking file names are the same
    pred_names_1 = sorted(pred_path_dict_1.keys())
    pred_names_2 = sorted(pred_path_dict_2.keys())

    print(len(pred_names_1), len(pred_names_2))
    print(pred_names_1[:20])
    print(pred_names_2[:20])
    assert pred_names_1 == pred_names_2

    n_files = len(pred_names_1)
    n_errors = 0
    n_successes = 0

    #
    # this zip is superfluous after this sorting step
    for file_1, file_2 in zip(pred_names_1, pred_names_2):

        path_1 = pred_path_dict_1[file_1]
        path_2 = pred_path_dict_2[file_2]

        print('Comparing files: {0} <> {1}'.format(path_1, path_2))

        if same_predictors(path_1, path_2):
            print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tOK')
            n_successes += 1
        else:
            print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tERROR')
            n_errors += 1

    #
    # ending
    print('\n\nComparisons ended.')
    print('Successes: {0}/{1}'.format(n_successes, n_files))
    print('Errors: {0}/{1}'.format(n_errors, n_files))

if __name__ == '__main__':

    DIR_1 = '/media/valerio/formalità/ultimate-scores/wiki/noneg/'
    DIR_2 = '/media/valerio/formalità/scores/ir_baseline/wiki/noneg/'

    # EXT_1 = 'te.*_(bm25|vsm|lm2|dfr2).*_([1-5]).scores'
    # EXT_2 = 'ts.*_(bm25|vsm|lm2|dfr2).*_([1-5]).scores'
    EXT_1 = 'tr.*_(bm25|vsm|lm2|dfr2).*_([1-5]).scores'
    EXT_2 = 'tr.*_(bm25|vsm|lm2|dfr2).*_([1-5]).scores'

    PATTERN_1 = 'tr_'
    PATTERN_2 = 'tr_'
    # PATTERN_2 = 'ts_'

    # PATTERN_1 = 'tr_'
    # PATTERN_2 = 'tr_neg_'

    compare_predictors_dirs(DIR_1, DIR_2, EXT_1, EXT_2, PATTERN_1, PATTERN_2)

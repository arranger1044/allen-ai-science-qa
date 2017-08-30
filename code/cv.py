from sklearn.cross_validation import KFold


def fold_gen(data_matrix, target, n_folds, shuffle=True, seed=1337):

    n_observations = data_matrix.shape[0]
    n_features = data_matrix.shape[1]
    print('Data matrix with {0} X {1}'.format(n_observations, n_features))

    #
    # getting fold ids
    k_fold_iter = KFold(n_observations, n_folds, shuffle=shuffle, random_state=seed)

    for i, (train_ids, test_ids) in enumerate(k_fold_iter):
        print('FOLD #{0}: Train size: {1} test size: {2}'.format(i,
                                                                 train_ids.shape[0],
                                                                 test_ids.shape[0]))
        yield (data_matrix[train_ids], target[train_ids]), \
            (data_matrix[test_ids], target[test_ids])

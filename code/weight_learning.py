import numpy

import theano
# import theano.config

import climate

climate.enable_default_logging()

import downhill

from time import perf_counter


def weighted_sum(preds_tensor, weights):
    P = theano.tensor.sum(preds_tensor * weights.dimshuffle('x', 'x', 0), axis=2)
    return P


def logistic_loss(preds_tensor, weights):
    P = theano.tensor.sum(preds_tensor * weights.dimshuffle('x', 'x', 0), axis=2)
    L = theano.tensor.nnet.softmax(P)
    return L


def logistic_loss_hada(preds_tensor, weights):
    P = theano.tensor.sum(preds_tensor * weights .dimshuffle(0, 'x', 1), axis=2)
    L = theano.tensor.nnet.softmax(P)
    return L


def logistic_loss_sm(preds_tensor, weights):
    weights = theano.tensor.nnet.softmax(weights.dimshuffle('x', 0))
    P = theano.tensor.sum(preds_tensor * weights.dimshuffle('x', 0, 1), axis=2)
    L = theano.tensor.nnet.softmax(P)
    return L


def neg_loglikelihood(preds, y):
    return -theano.tensor.mean(theano.tensor.log(preds)[theano.tensor.arange(y.shape[0]), y])


def cross_entropy(preds, y):
    return theano.tensor.nnet.categorical_crossentropy(preds, y).mean()


def hard_preds(preds):
    return theano.tensor.argmax(preds, axis=1)


def error(preds, actual_preds):
    return theano.tensor.mean(theano.tensor.neq(preds, actual_preds))


def accuracy(preds, actual_preds):
    return theano.tensor.mean(theano.tensor.eq(preds, actual_preds))


def test_logistic(preds_matrix, answers):

    P = theano.shared(value=preds_matrix, name='P', borrow=True)
    w = theano.shared(value=numpy.ones(preds_matrix.shape[2]), name='w')

    y = theano.shared(value=answers, name='y')

    # ensemble_output = hard_preds(weighted_sum(P, w))
    ensemble_output = hard_preds(logistic_loss(P, w))

    loss = error(ensemble_output, y)

    loss_func = theano.function([], loss)

    acc = loss_func()

    return acc


def test_logistic_hada(preds_matrix, answers):

    P = theano.shared(value=preds_matrix, name='P', borrow=True)
    w = theano.shared(value=numpy.ones((preds_matrix.shape[0],
                                        preds_matrix.shape[2])), name='w')

    y = theano.shared(value=answers, name='y')

    # ensemble_output = hard_preds(weighted_sum(P, w))
    ensemble_output = hard_preds(logistic_loss_hada(P, w))

    loss = error(ensemble_output, y)

    loss_func = theano.function([], loss)

    acc = loss_func()

    return acc


def test_downhill(preds_matrix, answers, learning_rate=0.6):
    # w = theano.shared(value=numpy.random.rand(preds_matrix.shape[2]), name='w')
    w = theano.shared(value=numpy.ones(preds_matrix.shape[2]) / preds_matrix.shape[2], name='w')
    P = theano.tensor.dtensor3('P')
    y = theano.tensor.lvector('y')
    alpha = 0.0
    beta = 0.2

    # loss = error(hard_preds(logistic_loss(P, w)), y)
    # loss = neg_loglikelihood(logistic_loss(P, w), y) + alpha * \
    #    (w * w).mean() + beta * abs(w).mean()
    loss = cross_entropy(logistic_loss(P, w), y) + alpha * \
        (w * w).mean() + beta * abs(w).mean()

    batch_size = 250

    err = accuracy(hard_preds(logistic_loss(P, w)), y)

    downhill.minimize(
        loss,
        [preds_matrix[2000:], answers[2000:]],
        valid=[preds_matrix[:500], answers[:500]],
        inputs=[P, y],
        patience=20,
        # batch_size=batch_size,
        algo='adadelta',
        # momentum=0.1,
        # max_gradient_norm=1,
        learning_rate=learning_rate,
        monitors=(('loss', loss),
                  ('err', err)),
        monitor_gradients=True)

    print(w.get_value())


def test_downhill_hada(preds_matrix, answers, learning_rate=0.6):
    # w = theano.shared(value=numpy.random.rand(preds_matrix.shape[2]), name='w')
    w = theano.shared(value=numpy.ones((preds_matrix.shape[0],
                                        preds_matrix.shape[2])) / preds_matrix.shape[2], name='w')
    print(w.shape)
    P = theano.tensor.dtensor3('P')
    y = theano.tensor.lvector('y')
    alpha = 0.0
    beta = 0.2

    # loss = error(hard_preds(logistic_loss(P, w)), y)
    # loss = neg_loglikelihood(logistic_loss(P, w), y) + alpha * \
    #    (w * w).mean() + beta * abs(w).mean()
    loss = cross_entropy(logistic_loss_hada(P, w), y) \
        # + alpha *  (w * w).mean() + beta * abs(w).mean()

    batch_size = 250

    err = accuracy(hard_preds(logistic_loss_hada(P, w)), y)

    downhill.minimize(
        loss,
        [preds_matrix[2000:], answers[2000:]],
        valid=[preds_matrix[:500], answers[:500]],
        inputs=[P, y],
        patience=20,
        # batch_size=batch_size,
        algo='adadelta',
        # momentum=0.1,
        # max_gradient_norm=1,
        learning_rate=learning_rate,
        monitors=(('loss', loss),
                  ('err', err)),
        monitor_gradients=True)

    print(w.get_value())


def test_theano(preds_matrix, answers, batch_size=100,
                learning_rate=0.01, n_epochs=100):

    # train_set_x, train_set_y = datasets[0]
    train_set_x = theano.shared(preds_matrix)
    train_set_y = theano.shared(answers)
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = theano.tensor.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = theano.tensor.dtensor3('X')  # data, presented as rasterized images
    y = theano.tensor.lvector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    # cost = classifier.negative_log_likelihood(y)
    w = theano.shared(value=numpy.random.rand(preds_matrix.shape[2]), name='w', borrow=True)
    # cost = error(hard_preds(logistic_loss(x, w)), y)
    cost = neg_loglikelihood(logistic_loss(x, w), y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    # test_model = theano.function(
    #     inputs=[index],
    #     outputs=errors(y),
    #     givens={
    #         x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: test_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    # validate_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = theano.tensor.grad(cost=cost, wrt=w)
    # g_b = theano.tensor.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(w, w - learning_rate * g_W)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = perf_counter()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print('\nEpoch', epoch)
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            print(iter, minibatch_avg_cost)
            print(w.get_value())

            # if (iter + 1) % validation_frequency == 0:
            #     # compute zero-one loss on validation set
            #     validation_losses = [validate_model(i)
            #                          for i in xrange(n_valid_batches)]
            #     this_validation_loss = numpy.mean(validation_losses)

            #     print(
            #         'epoch %i, minibatch %i/%i, validation error %f %%' %
            #         (
            #             epoch,
            #             minibatch_index + 1,
            #             n_train_batches,
            #             this_validation_loss * 100.
            #         )
            #     )

            #     # if we got the best validation score until now
            #     if this_validation_loss < best_validation_loss:
            #         # improve patience if loss improvement is good enough
            #         if this_validation_loss < best_validation_loss *  \
            #            improvement_threshold:
            #             patience = max(patience, iter * patience_increase)

            #         best_validation_loss = this_validation_loss
            #         # test it on the test set

            #         test_losses = [test_model(i)
            #                        for i in xrange(n_test_batches)]
            #         test_score = numpy.mean(test_losses)

            #         print(
            #             (
            #                 '     epoch %i, minibatch %i/%i, test error of'
            #                 ' best model %f %%'
            #             ) %
            #             (
            #                 epoch,
            #                 minibatch_index + 1,
            #                 n_train_batches,
            #                 test_score * 100.
            #             )
            #         )

            #         # save the best model
            #         with open('best_model.pkl', 'w') as f:
            #             cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = perf_counter()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') % (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))


if __name__ == '__main__':
    from dataset import load_data_frame
    from dataset import get_answers

    train_data = load_data_frame('../data/training_set.tsv')
    train_answers = get_answers(train_data, numeric=True)

    from ensemble import collect_predictions_from_dirs

    dirs = ['../scores/ir_baseline/1201/noneg/noneg_1201/',
            '../scores/ir_baseline/1201/neg/neg_1201/']
    train_pattern = 'train.scores|tr.*_(bm25|vsm|lm2|dfr2).*_([1-9]|10).scores'
    preds = collect_predictions_from_dirs(dirs,
                                          train_pattern)
    print(preds.shape)

    from weight_learning import test_logistic
    from weight_learning import test_downhill
    from weight_learning import test_theano

    lhp = test_logistic_hada(preds, train_answers)
    print(lhp)

    test_downhill(preds, train_answers)

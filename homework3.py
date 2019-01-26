# Author: Sarim Khan

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

# comments
# convert y into indicator matrix
# N is the size of dataset
# K is the number of class


def y2indicator(y):
    """

    :param y: rank-one target array of size N
    :return: N X K indicator matrix for target
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# comments
# calculate the error rate
# p is the prediction vector
# t is the target vector


def error_rate(p, t):
    """

    :param t: vector of integers
    :param p: vector of integers
    :return: error_rate, which is float value

    This function gives us the error rate between targets and predictions
    """
    # shape of targets(1000, )
    # shape of predictions(1000, )
    # print("shape of targets", targets.shape)
    # print("shape of predictions", predictions.shape)
    return np.mean(t != p)


def flatten(X):
    """

    :param X: input will be (32, 32, 3, N) basically input is 4d matrix
    :return: output will be (N, 3072) 2d numpy matrix
    """

    N = X.shape[-1] # matlab has number of rows as the last dimension
    flat = np.zeros((N, 3072)) # 3072 = 32x32x3
    for i in range(N):
        flat[i] = X[:, :, :, i].reshape(3072)
    return flat


def main():

    # reading the train and test data
    # we do not need split data into train and test since it is already done for us
    #
    train = loadmat('large_files/train_32x32.mat')
    test = loadmat('large_files/test_32x32.mat')

    print('type of train', type(train))
    print('type of train[X]', type(train['X']))

    # Need to scale training set! Don't leave as 0.255
    # Y is a N x 1 matrix with values 1 .. 10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    # flatten means combining several dimensions into one dimension

    # each element of 4d vector is real number
    Xtrain = flatten(train['X'].astype(np.float32)/255)

    # Ytrain is rank-one array
    Ytrain = train['y'].flatten() - 1  # subtract one from all elements

    print('shape of train[y] before flatten', train['y'].shape)
    print('shape of Ytrain after flatten', Ytrain.shape)

    # we shuffle it so that we can get different results each time
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)
    N, D = Xtrain.shape

    Xtest = flatten(test['X'].astype(np.float32)/255)
    Ytest = test['y'].flatten() - 1 # Ytest is rank-one array
    Ytest_ind = y2indicator(Ytest)

    # comments
    # maximum number of iteration (max_iter) is 20
    # batch size (batch_sz) is 500
    # You need to figure out the number of batches (n_batches)
    max_iter = 20
    batch_sz = 500
    n_batches = N // batch_sz

    # comments
    # we will use two hidden layer architecture
    # first hidden layer (M1) has 1000 hidden units
    # second hidden layer (M2) has 500 hidden units
    # Number of class (K) is 10
    # use np.random.randn to initialize the W1_init, W2_init, and W3_init
    # use np.zeros to initialize b1_init, b2_init, and b3_init
    #
    # W1_init is used to initialize the value of tensorflow variable W1
    M1 = 1000
    M2 = 500
    K = 10
    W1_init = np.random.randn(D, M1) / 28
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b1_init = np.zeros(M1)
    b2_init = np.zeros(M2)
    b3_init = np.zeros(K)

    # Comments
    # everything in tensorflow is float32
    # define X as tf.placeholder with datatype tf.float32 and its shape is (None, D)
    # X is the input data
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')

    # Define T as tf.placeholder with datatype tf.float32 and its shape is (None, K)
    # T is the target
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    # define W1 as tf.Variable and use W1_init.astype(np.float32) as its initial value
    W1 = tf.Variable(W1_init.astype(np.float32))

    # define b1 as tf.Variable and use b1_init.astype(np.float32) as its initial value
    b1 = tf.Variable(b1_init.astype(np.float32))

    # similarly define tf.Variable including W2, b2, W3, and b3
    #
    W2 = tf.Variable(W2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # comments
    # use tensorflow function to define A1, which is first hidden output layer
    # define A2, which is second hidden layer output
    # for A2 and A3, please use tf.nn.relu as the activation function
    # define Z3, which is the output of the output layer before passing through softmax
    # remember, the cost function does the softmaxing
    #
    A1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
    A3 = tf.nn.relu(tf.matmul(A2, W3) + b3)
    Z3 = tf.matmul(A2, W3) + b3

    # Comments
    # Define cost using tf.reduce_sum and tf.nn.softmax_cross_entropy_with_logits
    #

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=A3,
            labels=T
        )
    )

    # Comments
    # define train_op using tf.train.RMSPropOptimizer
    # learning rate is 0.0001
    # decay = 0.99
    # momentum = 0.9
    # we choose optimizer but don't implement the algorithm ourselves
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # comments
    # define predict_op to produce prediction results
    # use tf.argmax(Z3, 1)
    #
    predict_op = tf.argmax(Z3, 1)

    # Comments
    # use one batch to check the dimensionality of output value for hidden layers and output layer
    #
    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = Ytrain_ind[j*batch_sz:(j * batch_sz + batch_sz), ]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                # comments
                # use batch gradient descent to train neural networks
                # for every 10 batches, print the test cost and test error using testing set
                # Finally, draw the graph for testing cost
                #

                if j % 10 == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("i: ", i, "test cost: ", test_cost, "test error: ", err)
                    costs.append(test_cost)

    print("Final accuracy: ", error_rate(Ytrain, predict_op))
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()


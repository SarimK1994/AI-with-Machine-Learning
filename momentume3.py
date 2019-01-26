# Name:
# Date:
# Purpose: implement one layer neural network using momentum and RMSprop
# we will use neural network to recognize digits for MNIST datasets
#

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from utilMNIST import get_normalized_data, error_rate, cost, y2indicator
from mlpv2 import forward, derivative_w2, derivative_w1, derivative_b1, derivative_b2


def main():
    X, Y = get_normalized_data()
    lr = 0.00004 # pre-computed learning_rate
    reg = 0.01 # pre-computed regularization parameter

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    Ytrain_ind = y2indicator(Ytrain)  # get indicator matrix
    Ytest_ind = y2indicator(Ytest)

    print("size of Ytrain_ind", Ytrain_ind.shape)
    print("size of Ytest_ind", Ytest_ind.shape)
    print("size of Xtest", Xtest.shape)
    print("size of Ytest_ind", Ytest_ind.shape)

    N, D = Xtrain.shape
    batch_sz = 500 # how many samples are used for each minibatch
    n_batches = N // batch_sz # number of batches for training set

    M = 300 # 300 hidden unit in first hidden layer
    K = 10
    # weight initialization
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros((1, M))
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros((1, K))

    # save initial weights
    # return a copy of the array
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()


    # minibatch gradient descent with momentum
    # re-initialize W1, b1, W2, b2

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    losses_momentum = []
    errors_momentum = []

    mu = 0.9  # momentum parameter. mu is beta term

    max_iter = 30 # 30 epochs
    print_period = 30

    # these are previous weight changes
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0

    print("mini batch momentum")

    for i in range(max_iter):
        for j in range(n_batches):
            # Xbatch represents 500 training sample for one batch
            # Ybatch represents 500 corresponding target for one batch
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz)]
            # here A2 is 500x10 matrix. A1 is (500,300)
            A2,A1 = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(A1, Ybatch, A2) + reg*W2
            gb2 = derivative_b2(Ybatch, A2) + reg*b2
            gW1 = derivative_w1(Xbatch, A1, Ybatch, A2, W2) + reg*W1
            gb1 = derivative_b1(A1, Ybatch, A2, W2) + reg*b1

            # update velocities
            dW2 = mu*dW2 - lr*gW2
            db2 = mu*db2 - lr*gb2
            dW1 = mu*dW1 - lr*gW1
            db1 = mu*db1 - lr*gb1

            W2 += dW2
            b2 += db2
            W1 += dW1
            b1 += db1

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l1 = cost(pY, Ytest_ind)
                losses_momentum.append(l1)

                err = error_rate(pY, Ytest)
                print("Momentum iteration:", i, "batch:", j, "cost:", l1, "error:", err)

    #  Outside of double for loop
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate for momentum batch:", error_rate(pY, Ytest))
    print()
    print()

    print("RMSprop")
    # reinitialize the weight
    # reinitialize W1, b1, W2, and b2
    # I want momentum and RMSprop to start with same weight for fair comparison
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    LL_rms = []

    cache_W2 = 1 # exponentially weighted average for dW2
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1

    decay_rate = 0.999
    eps = 1e-10

    for i in range(max_iter):
        for j in range(n_batches):
            #  Mini-batch for whole training set

            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz)]
            A2, A1 = forward(Xbatch, W1, b1, W2, b2)

            lr0 = 0.001

           #  calculate gradient of cost function
            gW2 = derivative_w2(A1, Ybatch, A2) + reg * W2
            cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
            W2 = W2 - lr0 * gW2 / (np.sqrt(cache_W2) + eps)

            gb2 = derivative_b2(Ybatch, A2) + reg * b2
            cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
            b2 = b2 - lr0 * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(Xbatch, A1, Ybatch, A2, W2) + reg * W1
            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
            W1 = W1 - lr0 * gW1 / (np.sqrt(cache_W1) +eps)

            gb1 = derivative_b1(A1, Ybatch, A2, W2) + reg * b1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
            b1 = b1 - lr0 * gb1 / (np.sqrt(cache_b1) + eps)

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l1 = cost(pY, Ytest_ind)
                LL_rms.append(l1)

                err = error_rate(pY, Ytest)
                print("RMS iteration:", i, "batch:", j, "cost:", l1, "error:", err)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error for RMSprop: ", error_rate(pY, Ytest))
    print()
    print()

    print("Mini batch gradient descent without momentum: ")
    N, D = Xtrain.shape
    batch_sz = 500 # how many samples are used for each minibatch
    n_batches = N // batch_sz # number of batches for training set

    M = 300 # 300 hidden unit in first hidden layer
    K = 10
    # weight initialization
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros((1, M))
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros((1, K))

    # save initial weights
    # return a copy of the array
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()


    # minibatch gradient descent with momentum
    # re-initialize W1, b1, W2, b2

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    miniBatch = []

    max_iter = 30 # 30 epochs
    print_period = 30

    # these are previous weight changes
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0

    print("minibatch without momentum")

    for i in range(max_iter):
        for j in range(n_batches):
            # Xbatch represents 500 training sample for one batch
            # Ybatch represents 500 corresponding target for one batch
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz)]
            # here A2 is 500x10 matrix. A1 is (500,300)
            A2,A1 = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(A1, Ybatch, A2) + reg*W2
            gb2 = derivative_b2(Ybatch, A2) + reg*b2
            gW1 = derivative_w1(Xbatch, A1, Ybatch, A2, W2) + reg*W1
            gb1 = derivative_b1(A1, Ybatch, A2, W2) + reg*b1

            # update velocities
            W2 = W2 - lr*gW2
            b2 = b2 - lr*gb2
            W1 = W1 - lr*gW1
            b1 = b1 - lr*gb1

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l1 = cost(pY, Ytest_ind)
                miniBatch.append(l1)

                err = error_rate(pY, Ytest)
                print("minibatch iteration:", i, "batch:", j, "cost:", l1, "error:", err)

    #  Outside of double for loop
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate for no momentum batch:", error_rate(pY, Ytest))

    # plot three cost in the plot to compare them
    plt.plot(losses_momentum, label="momentum")
    plt.plot(LL_rms, label="rms")
    plt.plot(miniBatch, label="minibatchNoMomentum")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

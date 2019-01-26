# updated 4/14/2018
# This is Utilities functions for facial recognition of Logistic regression
# Y is N x 1 vector
#

import numpy as np
import pandas as pd
from sklearn.utils  import shuffle

# x > 0 will return 1 or 0
#
def relu(x):
    return x * (x > 0)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))


# add for logistic softmax
#
def softmax(A):
    """

    :param a: N x K prediction matrix 
    :return:  N x K normalized prediction matrix 
    """
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    """
    
    :param T: vector for target
    :param Y: vector for real value prediction
    :return: cross-entropy error, which is float
    
    calculates the cross entropy from the definition for sigmoid cost.
    This is used for binary classification
    """
    # shape of T in sigmoid_cost(1000, 1)
    # shape of Y in sigmoid_cost(1000, 1)
    # print("shape of T in sigmoid_cost", T.shape)
    # print("shape of Y in sigmoid_cost", Y.shape)
    # all operations are element-wise operation
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# add for logistic softmax
# this is softmax cost function
#
def cost(T,Y):
    """

       :param T: N x K indicator matrix
       :param Y: N x K normalized prediction matrix
       :return: cost

       Note: * is element-wise operation
       definition = target * log(Y)
       this is more general cost function, which works for softmax
       """
    return -(T*np.log(Y)).sum()

# add for logistic softmax
# this is softmax cost
# ?
def cost2(T,Y):
    """
    
    :param T: N x 1 target row vector
    :param Y: N x K normalized prediction matrix
    :return: cost
    """
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    # select all logs which matches the target
    #
    # shape of T in cost2(1000, 1)
    # shape of Y in cost2(1000, 7)
    # value of N in cost2 1000

    N = len(T)
    # print("shape of T in cost2", T.shape)
    # print("shape of Y in cost2", Y.shape)
    # print("value of N in cost2", N)
    # a = T[:,0]
    # print("shape of a", a)
    return -np.log(Y[np.arange(N), T[:,0]]).mean()


def error_rate(targets, predictions):
    """
    
    :param targets: vector of integers
    :param predictions: vector of integers
    :return: error_rate, which is float value
    
    This function gives us the error rate between targets and predictions
    """
    # shape of target in error rate(1000, 1)
    # shape of predictions in error rate(1000, 1)
    # print("shape of target in error rate", targets.shape)
    # print("shape of predictions in error rate", predictions.shape)
    return np.mean(targets != predictions)


# add for logistic softmax
# convert the target vector into N x K indicator matrix
#
def y2indicator(y):
    """

    :param y: N x 1 column vector representing the target for each sample
    :return: N x K indicator matrix
    """
    # shape of y in y2indicator (39263, 1)
    # print("shape of y in y2indicator", y.shape)
    N = len(y)
    K = len(set(y[:,0]))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i,0]] = 1
    return ind


def getData(balance_ones=True):
    """

    :param balance_ones: 
    :return: X, which is 2d numpy array   Y, which is target and column vector

    this function will get all the data from all the classes
    """

    # images are 48x48 = 2304 size vectors
    # N = 35887
    # We need skip the first line which is just header
    # first column is label and second column is space separated pixels
    # third column is usage
    # we turn all of these into integers
    #
    Y = []
    X = []
    first = True

    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    print('finish reading all data in the get_data function')

    # normalize X, which goes from 0 to 1 instead of 0 to 255
    # X is 2d numpy array. Each row is one sample
    #
    X, Y = np.array(X) / 255.0, np.array(Y)
    Y = np.reshape(Y, (Y.shape[0], 1))  # changed
    # shape of X before data balancing (35887, 2304)
    # shape of Y before data balancing(35887, )
    # print('shape of X before data balancing', X.shape)
    # print('shape of Y before data balancing', Y.shape)

    if balance_ones:
        # balance the class 1
        # X0 is all the sample which is not class 1
        # get column 0 for Y
        # shape Y != 1 (35887, 1)
        # shape X0 (35340, 2304)
        # shape Y0 (35340, 1)
        # shape X1 (547, 2304)
        X0 = X[Y[:, 0] != 1]
        Y0 = Y[Y[:, 0] != 1]
        X1 = X[Y[:, 0] == 1]

        # print("shape X0", X0.shape)
        # print("shape Y0", Y0.shape)
        # print("shape X1", X1.shape)
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])  # stack row-wise

        # shape X1 after repeat (4923, 2304)
        # shape X (40263, 2304)
        # print("shape X1 after repeat", X1.shape)
        # print("shape X", X.shape)
        temp = np.array([1] * len(X1))
        temp = np.reshape(temp, (temp.shape[0], 1))
        Y = np.concatenate((Y0, temp), axis=0)
        # print("shape Y", Y.shape)

        # shape of X (40263, 2304)
        # shape of Y (40263, 1)

    return X, Y


def getBinaryData():
    """
    :return: 2d matrix X for data, vector Y for target

    Notes: get data from class 0 and class 1 only. We do not deal with class
    imbalance problem here
    """
    Y = []
    X = []
    first = True

    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])  # get the label
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    X = np.array(X) / 255.0  # changed
    Y = np.array(Y)  # changed
    Y = np.reshape(Y, (Y.shape[0], 1))  # changed
    return X, Y  # changed


#
# comments
# this is homework 1
# function used for crossValidation
#
def crossValidation(model, X, Y, K=5):
    """

    :param model: initial logistic model
    :param X: 2d array of input data
    :param Y: vector of target
    :param K: number of folds
    :return: vector of accuracy for K fold

    using K-1 folds for training to produce model
    using remaining one fold for testing
    """
    # comments
    # shuffle X and Y
    #
    X, Y = shuffle(X, Y)

    # comments
    # calculate size of one fold
    #
    sz = int(len(Y) / K)
    # size of one fold: 3292
    print("size of one fold: " + str(sz))

    # comments
    # initialize accs, which is list of accuracy for each fold
    #
    accs = []

    # comments
    # working on fold by fold
    #
    for k in range(K):
        # comments
        # getting all k-1 folds for training (xtr, ytr)
        # getting one fold for crossvalidation (xte, yte)
        # np.concatenate joins a sequence of arrays along an existing axis
        #
        xtr = np.concatenate([X[:k * sz], X[(k * sz + sz):]], axis=0)
        ytr = np.concatenate([Y[:k * sz], Y[(k * sz + sz):]], axis=0)
        xte = X[k * sz:(k * sz + sz)]
        yte = Y[k * sz:(k * sz + sz)]

        # shape of xtr in CV(6584, 2304)
        # shape of ytr in CV(6584, 1)
        # shape of xte in CV(3292, 2304)
        # shape of yte in CV(3292, 1)

        print()
        print('shape of xtr in CV', xtr.shape)
        print('shape of ytr in CV', ytr.shape)
        print('shape of xte in CV', xte.shape)
        print('shape of yte in CV', yte.shape)
        print()

        # comments
        # training the model using one fold
        #
        model.fit(xtr, ytr, show_fig=True)

        # comments
        # getting the accuracy for one fold
        # print out accuracy for one fold
        #
        acc = model.score(xte, yte)
        print('fold: ', k, ' acc: ', acc)

        # comments
        # append acc to accs list
        #
        accs.append(acc)

    # comments
    # return accs list
    #
    return accs


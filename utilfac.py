# updated 12/23/2017
#

import numpy as np
import pandas as pd
from sklearn.utils  import shuffle

def init_weight_and_bias(M1, M2):
    """
    
    :param M1: number of hidden unit in input layer
    :param M2: number of hidden unit in output layer
    :return: weight matrix and bias vector
    """
    W = np.random.randn(M1, M2) / np.sqrt(M1)  # gaussian distributed with small variance
    b = np.zeros(M2)
    # We want to turn these into float32 so that we can use them in Theano and tensor flow
    # without them complaining to us
    #
    return W.astype(np.float32), b.astype(np.float32)


# theano version
# (num_feature_maps, old_num_feature_maps, filter_width, filter_height)
#
# tensor flow version
# (filter width, filter height, input feature maps, output feature maps)
# weight initialization which is based on dimensions
#
# this function is used for theano version only
# The * unpacks a tuple into multiple input arguments.
# shape is the tuple (3, 10, 5, 5)
# np.sqrt(fan_in total dimension(need multiply), fan_out total dimesnion)
def init_filter(shape, poolsz):
    # w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) +
    #                                              shape[0]*np.prod(shape[2:] / np.prod(poolsz)))

    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]))

    return w.astype(np.float32)

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
    
    calculates the cross entropy from the definition for sigmoid cost (error).
    so that it is binary classification
    """
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# add for logistic softmax
# this is softmax cost
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
#
def cost2(T,Y):
    """
    
    :param T: rank-one array for target
    :param Y: N x K normalized prediction matrix
    :return: cost
    """
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    # select all logs which matches the target
    #
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
    """
    
    :param targets: vector of integers
    :param predictions: vector of integers
    :return: error_rate, which is float value
    
    This function gives us the error rate between targets and predictions
    """
    # shape of targets(1000, )
    # shape of predictions(1000, )
    # print("shape of targets", targets.shape)
    # print("shape of predictions", predictions.shape)
    return np.mean(targets != predictions)


# add for logistic softmax
#
def y2indicator(y):
    """

    :param y: rank-one array representing the target for each sample
    :return: N x K indicator matrix
    """
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData(balance_ones=True):
    """
    
    :param balance_ones: 
    :return: N x D input matrix X and rank-one array for target Y
    
    this function will get all the data from all the classes
    """

    # images are 48x48 = 2304 size vectors
    # N = 35887
    # We need skip the first line which is just header
    # first column is label and second column is space separated pixels
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

    print('finish first step for getData')


    # normalize X, which goes from 0 to 1 instead of 0 to 255
    # X is 2d array. Each row is one sample
    #
    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class by repeating it 9 times
        # X0 is all the sample which is not class 1
        #
        X0, Y0 = X[Y !=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1,9,axis=0)
        X = np.vstack([X0,X1])         #stack row-wise
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


# this function keep the original image shape
def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)  # only have one color channel  48 x 48 image per sample
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
            y = int(row[0])       # get the label
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])

    return np.array(X) / 255.0, np.array(Y)

# verify
#
def crossValidation(model, X, Y, K=5):
    """
    
    :param model: initial logistic model
    :param X: 2d array of input data
    :param Y: vector of target
    :param K: number of folds
    :return: vector of accuracy for K fold
    
    using K-1 fold for training to produce model
    using remaining one fold for testing
    """

    # split data into K parts
    X, Y = shuffle(X,Y)
    sz = len(Y) // K   # size of one fold
    accs = []

    for k in range(K):
        # getting all k-1 folds for training
        #
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr, show_fig=True)
        acc = model.score(xte,yte)
        print('fold: ', k, ' acc: ', acc)
        accs.append(acc)

    return accs


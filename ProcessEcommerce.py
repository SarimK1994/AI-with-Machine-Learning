# 4/14/2018
# This file has several functions to preprocess the e-commmerce data
#

import numpy as np
import pandas as pd

# Column 0: Is_Mobile( 0/1)
# Column 1: N_Product_Viewed (int >= 0)
# Column 2: visit_during (real >= 0)
# Column 3: is_returning_visitor(0/1)
# Column 4: Time_of_day (0/1/2/3)


def get_data():
    df = pd.read_csv('ecommerce_data.csv')

    # just in case you're curious what's in it
    # print(df.head())

    # easier to work with numpy array
    data = df.as_matrix()

    # Separate data into 2 parts
    # X is training data and Y is target
    # X is (500, 5)
    # Y is (500, 1)

    X = data[:,:-1]
    Y = data[:,-1] # get last column
    Y = np.reshape(Y, (Y.shape[0], 1))

    # print("shape of X before processing", X.shape)
    # print("shape of Y before processing", Y.shape)

    # normalize numerical columns
    # normalize columns 1 and 2
    #
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # one-hot categorical columns which is Time_of_day (0/1/2/3)
    # create a new matrix X2 with the correct number of columns
    # I add three column and as a result, last four columns is used
    # for one-hot encoding.
    #
    # the categorical number: 0 1 2 3
    # if it is 1, one-hot encoding: 0 1 0 0
    # if it is 3, one-hot encoding: 0 0 0 1
    #
    N,D = X.shape
    X2 = np.zeros((N,D+3))

    # copy non-categorical column
    # note last column of X is categorical
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # one-hot encoding for last four column
    # this loop go through each sample
    for n in range(N):

        # last column of X is categorical features
        # we convert it into int
        # D-1 is starting of column for categorical column
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    # Another way to do one-hot encoding
    Z = np.zeros((N,4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:-4:] = Z
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    # shape of X2 (500, 8)
    # shape of Y (500, 1)
    # print("shape of X2", X2.shape)
    # print("shape of Y", Y.shape)

    return X2, Y


#
# return only the data from the first 2 classes
# target is 0 1 2 3
#
def get_binary_data():
    X, Y = get_data()
    # filter it by taking only class 0 and class 1
    # conditional selecting elements from the original array
    #  shape of X2 (398, 8)
    #   shape of Y (398, 1)
    # access index for each row of 2d nparray
    X2 = X[Y[:,0] <= 1]
    Y2 = Y[Y[:,0] <= 1]
    return X2, Y2


# X2,Y2 = get_binary_data()
# print()
# print("shape of X2", X2.shape)
# print(X2)
# print("shape of Y", Y2.shape)
# print(Y2)

def y2indicator(y):
    """

    :param y: N x 1 column vector representing the target for each sample
    :return: N x K indicator matrix  # which is one hot encoding
    """
    # shape of y in y2indicator (39263, 1)
    # print("shape of y in y2indicator", y.shape)
    N = len(y)
    K = len(set(y[:, 0]))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, int(y[i, 0])] = 1
    return ind

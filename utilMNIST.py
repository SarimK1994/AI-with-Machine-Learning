# updated 12/24/2017
# Some utility functions we need for the class

# the data we get is MNIST dataset with 10 classes for 10 digits

# benchmark_full uses all features to train logistic regression model
# benchmark_pca  uses top 300 feature for training logistic regression model

# each sample is 28 x 28 image flatten by 1 x 784 vector
# each element of file is pixel density
#
# perform PCA to get top 300 principal components, which is fed into logistic regression later
# Final error rate: 0.085 for pca
# Final error rate: 0.076 for full benchmark

# The PCA example shows that we only needed 300 features to be able to get comparable
# accuracy. This indicates that there is some linear transformation to a new vector of
# 300 that can get comparable accuracy.
# with nonlinear neural network with 300 hidden units, we should do just as good or perhaps
# even better than this
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_spiral():
    # Idea: radius -> low...high
    #           (don't start at 0, otherwise points will be "mushed" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ..., ]
    # x = rcos(theta), y = rsin(theta) as usual

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi*i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # convert into cartesian coordinates
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()

    # add noise
    X += np.random.randn(600, 2)*0.5

    # targets
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
    # We use rank-one array
    # Y = np.reshape(Y, (Y.shape[0], 1))
    print("shape of X in Sprial", X.shape)
    print("shape of Y in Sprial", Y.shape)
    return X, Y

# the data we get is MNIST dataset with 10 classes for 10 digits
# pca benchmark
# this will transform original data using PCA.
# We will keep all the data and no feature reduction is performed
def get_transformed_data():
    """
    
    :return pca-transformed 2D feature vector, vector of target, pca model with transformation matrix, 
            vector for mean in each dimension
    """
    print()
    print("Running get_transformed_data function")
    print()
    df = pd.read_csv('./large_files/train.csv')

    # type of data <class 'numpy.ndarray'>
    # shape of data(42000, 785)
    #
    data = df.as_matrix().astype(np.float32)  # data will be 2d numpy matrix
    print("type of data", type(data))
    print("shape of data", data.shape)
    np.random.shuffle(data)

    X = data[:, 1:]  # get data only

    # mu is the vector for each dimension.
    # axis=0 calculate mean across rows
    # shape of mu (1, 784)
    mu = X.mean(axis=0, keepdims=True)
    # print("shape of mu", mu.shape)

    # center the data
    # each row vector of X minus mu vector
    #
    X = X - mu
    pca = PCA()


    Z = pca.fit_transform(X)
    Y = data[:,0].astype(np.int32)  # Y is vector of label
    Y = np.reshape(Y, (Y.shape[0], 1))
    plot_cumulative_variance(pca)

    print("type of Z", type(Z))
    print("shape of Z", Z.shape)
    print("shape of Y", Y.shape)
    print("shape of mu", mu.shape)
    print()

    # type of Z <class 'numpy.ndarray'>
    # shape of Z (42000, 784)
    # shape of Y (42000, 1)
    # shape of mu (1, 784)

    return Z, Y, pca, mu


# normalize the data by subtracting mean and divided by standard deviation
#
def get_normalized_data():
    """
    
    :return: N x D matrix x and vector y of size N
    
    notes: normalize the input features
    """
    print()
    print("Running get_normalized data function")
    print()
    df = pd.read_csv('./large_files/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]

    # shape of mu(1, 784)
    # shape of std(1, 784)
    # shape of X(42000, 784)
    # shape of Y(42000, )
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)


    print("shape of mu", mu.shape)
    print("shape of std", std.shape)

    # this will replace the element having std value = 0 with std value = 1
    # Change elements of an array based on conditional and input values.
    np.place(std, std == 0, 1)
    X = (X - mu) / std  # normalize the data using broadcasting function
    Y = data[:,0]
    Y = np.reshape(Y, (Y.shape[0], 1))

    # shape of data (42000, 785) which includes the label
    # shape of X(42000, 784)
    # shape of Y(42000, 1)
    print("shape of X", X.shape)
    print("shape of Y", Y.shape)
    return X, Y


# function to plot cumulative variance after doing PCA
# pca benchmark
#
def plot_cumulative_variance(pca):
    """
    
    :param pca: pca model
    :return: list of cummulative variance calculated by PCA
    
    notes: this justifies taking top 300 principal components because it contains 
    over 95% of variance of the original data
    """
    # pca.explained_variance_ratio_ parameter returns a vector of the
    # variance explained by each dimension.
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            # append will add an item to the end of list
            # basically we are calculating cumulative_variance for each step
            #
            P.append(p + P[-1])

    plt.plot(P)
    plt.show()
    return P


# forward function for logistic regression using softmax
#
def forward(X, W, b):
    """
    
    :param X: 
    :param W: 
    :param b: 
    :return: N x K normalized prediction matrix
    
    notes: this is logisitc regression with softmax
    """
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y



def predict(p_y):
    """
    
    :param p_y: N x K normalized prediction matrix
    :return: prediction int vector with size N x 1
    """
    temp = np.argmax(p_y, axis=1)
    temp = np.reshape(temp, (temp.shape[0], 1))
    return temp


def error_rate(p_y, t):
    """
    
    :param p_y: N x K normalized prediction matrix
    :param t: target vector with size N x 1
    :return: 
    """
    prediction = predict(p_y)
    return np.mean(prediction != t)



def cost(p_y, t):
    """
    
    :param p_y: N x K normalized prediction matrix
    :param t: N x K indicator matrix
    :return: cost
    """
    # tot = t * np.log(p_y)  # element wise multiplication
    # return -tot.sum()

    # return -(t * np.log(p_y)).mean()
    # total cost is related to sum
    return -(t * np.log(p_y)).sum()



def gradW(t, y, X):
    """
    
    :param t: N x K indicator matrix
    :param y: N x K normalized prediction matrix
    :param X: N x D input
    :return: D x K gradient matrix for logistic softmax
    """
    return X.T.dot(y - t)


def gradb(t, y):
    return (y - t).sum(axis=0,keepdims=True)


# change y which is column vector of numbers from 0 to 9 into an indicator
# matrix N x 10
def y2indicator(y):
    """

    :param y: target vector of size N
    :return: N x K indicator matrix
    """
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i,0]] = 1
    return ind


# using all features to train logistic regression model
def benchmark_full():
    X, Y = get_normalized_data()
    print()
    print("Performing logistic regression using normalized data...")
    print("I am using all features to train the model")
    print()
    Xtrain = X[:-1000]  # filter based on rows
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros((1,10))
    LL = []  # cost for training data
    LLtest = []  # cost for testing data
    CRtest = []  # error for testing data

    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    lr = 0.00004  # learning rate

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error
    reg = 0.01

    for i in range(500):
        p_y = forward(Xtrain, W, b)
        # shape of p_y (41000, 10)
        # print("shape of p_y", p_y.shape)
        l1 = cost(p_y, Ytrain_ind)
        LL.append(l1)

        p_y_test = forward(Xtest, W, b)
        l1test = cost(p_y_test, Ytest_ind)
        LLtest.append(l1test)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W -= lr * (gradW(Ytrain_ind, p_y, Xtrain) + reg * W)
        b -= lr * (gradb(Ytrain_ind, p_y) + reg * b)

        if i % 20 == 0:
            print("iteration:", i, "training cost:", l1, "testing cost:", l1test, "error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()



# We only use top 300 feature for training logistic regression model
def benchmark_pca():
    """
    :return: 
    
    Notes: this will use 300 pca transformed features
    """
    # get_transformed_data return Z and Y, which is X and Y in this code
    X, Y, _, _ = get_transformed_data()
    X = X[:, :300] # get only 300 features

    # normalize X first based on each column or each feature
    # mu the mean vector for each column (feature)
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    X = (X - mu) / std

    print()
    print("I using 300 PCA selected features")
    print("Performing logistic regression...")
    print()
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:] # this is last 1000 samples
    Ytest  = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = np.zeros((N, 10)) # we have 0 - 9 classes
    for i in range(N):
        Ytrain_ind[i, Ytrain[i,0]] = 1

    Ntest = len(Ytest)
    Ytest_ind = np.zeros((Ntest, 10))
    for i in range(Ntest):
        Ytest_ind[i, Ytest[i,0]] = 1

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros((1,10))
    LL = []   # cost for training data
    LLtest = []  # cost for testing data
    CRtest = []  # error for testing data

    # D = 300 -> error = 0.07
    lr = 0.0001    # learning rate
    reg = 0.01

    for i in range(200):
        p_y = forward(Xtrain, W, b)
        l1 = cost(p_y, Ytrain_ind)
        LL.append(l1)

        p_y_test = forward(Xtest, W, b)
        l1test = cost(p_y_test, Ytest_ind)
        LLtest.append(l1test)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W -= lr*(gradW(Ytrain_ind, p_y, Xtrain) + reg*W)
        b -= lr*(gradb(Ytrain_ind, p_y) + reg*b)

        if i % 10 == 0:
            print("iteration:", i, "training cost:", l1, "testing cost:", l1test, "error rate:", err)
            #print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate for pca transformed feature:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest) # show cost for training and cost for testing
    plt.show()
    plt.plot(CRtest) # show error at testing set
    plt.show()

if __name__ == '__main__':
    # get_transformed_data()
    # benchmark_pca()
    benchmark_full()
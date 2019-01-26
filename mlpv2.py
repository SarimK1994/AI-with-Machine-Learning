# updated 3/18/2018
# this is used to implement momentum and rms
# Simple multi-layer perceptron / neural network in Python and Numpy
# this mlp has one hidden layer
# We have two versions for mlp including sigmoid and rectifier unit

import numpy as np

def forward(X, W1, b1, W2, b2):
    """
    
    :param X: 
    :param W1: 
    :param b1: 
    :param W2: 
    :param b2: 
    :return:  N x K normalized prediction matrix and N x M activation matrix
    """
    # sigmoid activation
    # Z = 1 / (1 + np.exp(-( X.dot(W1) + b1 )))

    # rectifier
    Z1 = X.dot(W1) + b1
    # shape of Z < 0 (500, 300)
    Z1[Z1 < 0] = 0
    A1 = Z1

    Z2 = A1.dot(W2) + b2
    expA = np.exp(Z2)
    A2 = expA / expA.sum(axis=1, keepdims=True)
    return A2, A1

# w2 is weight from hidden layer to output layer
def derivative_w2(A1, T, A2):
    """
    
    :param Z: 
    :param T: 
    :param Y: 
    :return: M x K derivative matrix for W2
    """
    return A1.T.dot(A2 - T)

# checked
def derivative_b2(T, A2):
    """
    
    :param T: 
    :param Y: 
    :return: derivative vector of size n for biased term
    """
    return (A2 - T).sum(axis=0, keepdims=True)

# checked
def derivative_w1(X, A1, T, A2, W2):
    # return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) )  # sigmoid
    return X.T.dot( ( (A2-T).dot(W2.T) * (A1 > 0) ) )   # relu

# checked
# Z > 0 will produce matrix of 1 or 0 depending on whether results are true or false
#
def derivative_b1(A1, T, A2, W2):
    # return (( Y-T ).dot(W2.T) * ( Z*(1-Z) )).sum(axis=0)  # sigmoid
    return ((A2-T).dot(W2.T) * (A1 > 0)).sum(axis=0, keepdims=True)  # relu

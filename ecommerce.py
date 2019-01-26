# Logistic Regression homework for Ecommerce Dataset
# this homework will practice how to use logistic regression to learn E-commerce data
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ProcessEcommerce import get_binary_data

# Comments
# call get_binary data to obtain training data X and its target Y
#
X, Y = get_binary_data()

# Comments
# shuffle training data X and its target Y in case that they are in order
#
X, Y = shuffle(X, Y)

# Comments
# print out the shape of X and Y
#
print("shape of X: ", X.shape)
print("shape of Y: ", Y.shape)

# Comments
# create training_set Xtrain, Ytrain, and test_set Xtest, Ytest
# last 100 rows will be test set and rest is training set
#
Xtrain, Ytrain = X[-100:], Y[-100:]
Xtest, Ytest = X[:-100], Y[:-100]

# Comments
# print out shape of Xtrain and Ytrain
#
print("shape of Xtrain: ", Xtrain.shape)
print("shape of Ytrain: ", Ytrain.shape)


# Comments
# randomly initialize weight vector W, which is D x 1, using np.random.randn
# D is the dimension of each sample in the dataset
# initialize the bias b to 0
#

N, D = X.shape
W = np.random.randn(D, 1) / np.sqrt(D)
b = 0

# Comments
# write a sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Comments
# Write the forward function
# X is NxD matrix
# W is Dx1 dimensional vector
# b is scalar
# return is Nx1 dimensional
#
def forward (X, W, b):
    """

     :param self:
     :param X: 2d array
     :return: N x 1 vector of prediction (Double)
     """

    # sigmoid normalizes the data
    # dot product is the .dot() and W is the weights
    return sigmoid(X.dot(W) + b)

# Comments
# calculate the accuracy
# Y is target vector and P is prediction vector
#
def classification_rate(Y, P):
    """

    :param targets: vector of integers
    :param predictions: vector of integers
    :return: error_rate, which is float value

    This function gives us the error rate between targets and predictions
    """
    # shape of target in error rate(1000, 1)
    # shape of predictions in error rate(1000, 1)
    # print("shape of target in error rate", Y.shape)
    # print("shape of predictions in error rate", P.shape)
    return np.mean(Y == np.round(P))

# Comments
# write cross entropy function
# T is target vector and pY is prediction vector
#please use np.sum()
#np.sum will return one number
#
def cross_entropy(T, pY):
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
    # print("shape of Y in sigmoid_cost", P.shape)
    # all operations are element-wise operation
    return -np.sum(T * np.log(pY) + (1 - T) * np.log(1 - pY))


# Comments
# initialize train_costs to empty list
# initialize test_costs to empty list
#
train_costs = []
test_costs = []

# Comments
# learning_rate is 0.001
#
learning_rate = 0.001

# Comments
# we enter the train loop which has 10000 iteration
#
for i in range(10000):

    # Comments
    # call forward function on Xtrain to get pYtrain, which is prediction vector for Xtrain
    #
    pYtrain = forward(Xtrain, W, b)

    # comments
    # call forward function on Xtest to get pYtest, which is prediction vector Xtest
    #
    pYtest = forward(Xtest, W, b)

    # Comments
    # call cross entropy on pYtrain to obtain training cost
    #
    cTrain = cross_entropy(Ytrain, pYtrain)

    # Comments
    # call cross entropy on pYtest to obtain test cost
    cTest = cross_entropy(Ytest, pYtest)

    # Comments
    # append training cost to train_costs list
    # append test cost to test_costs list
    #
    train_costs.append(cTrain)
    test_costs.append(cTest)

    # Comments
    # gradient descent (vectorized implementation)
    # Use the gradient of W and b to update weight vector W and bias b

    # gradient descent step
    # dZ is n x 1 vector
    # pY is n x 1
    # - is element-wise operation
    # dZ change in z with respect to L in theory dz=(dL/dZ) back in theory (math notes)
    # reg*self.W is regularization term to prevent weight becoming too big
    dZ = pYtrain - Ytrain  # Y?
    # one step of gradient descent
    # every iteration change W a little bit until reaching the global minimum
    W = W - learning_rate * (Xtrain.T.dot(dZ))
    b = b - learning_rate * (dZ.sum())

    # Comments
    # print out ctrain (training cost) and ctest (testing cost) every 1000 iteration
    #
    if i % 1000 == 0:
        print('Training cost: ', cTrain, "Testing Cost: ", cTest)

# Comments
# calculate final training accuracy and test accuracy after the training loop
#
e = classification_rate(Ytrain, pYtrain)
print("Error rate: ", e)


# Comments
# draw the graph for training cost and test cost
#
legend1, = plt.plot(train_costs, label='train_cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
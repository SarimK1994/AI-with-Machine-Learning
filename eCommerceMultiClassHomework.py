# Name
# Author
# Date

# This is one-hidden layer neural network for facial recognition
# We will use softmax for multi-class classification
# Warning: this program requires much more RAM than your previous files since we will use all classes

import numpy as np
import matplotlib.pyplot as plt
from ANN1FacialUtil import softmax, cost2, error_rate, relu
from sklearn.utils import shuffle
from ProcessEcommerce import get_data


def forward(X, W1, b1, W2, b2):
    """

    :param X: N x D matrix
    :return: N x K matrix normalized matrix (A2) and N x M activation matrix for first
             hidden layer
    """

    Z1 = X.dot(W1) + b1
    A1 = np.tanh(Z1)  # A1 is the activation value for first hidden layer using tanh function

    #  A2 is the activation for the output layer (prediction value)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)  # softmax is used for multi-class classification
    return A2, A1


def y2indicator(y, K):
    """

    :param y: N x 1 column vector representing the target for each sample
    :return: N x K indicator matrix  # which is one hot encoding
    """
    # shape of y in y2indicator (39263, 1)
    # print("shape of y in y2indicator", y.shape)
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, int(y[i, 0])] = 1
    return ind


def predict(X, W1, b1, W2, b2):
    """

    :param X: NxK input matrix
    :return: Nx1 prediction vector
    """
    pY, _ = forward(X, W1, b1, W2, b2)
    pred = np.argmax(pY, axis=1)
    pred = np.reshape(pred, (pred.shape[0], 1))
    return pred


def score(X, Y, W1, b1, W2, b2):
    """

    :param X: training set which is 2d array
    :param Y: vector of target
    :return: accuracy of model
    """
    prediction = predict(X, W1, b1, W2, b2)
    return 1 - error_rate(Y, prediction)


X, Y = get_data()
X, Y = shuffle(X, Y)


Xtrain, Ytrain = X[-100:], Y[-100:]
Xtest, Ytest = X[:-100], Y[:-100]

N, D = X.shape
K = len(set(Y[:, 0]))
T = y2indicator(Y, K)
M = 20
epochs = 10000
learning_rate = 0.001
reg = 0


W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros((1, M))

W2 = np.random.randn(M, K) / np.sqrt(M)
b2 = np.zeros((1, K))
trainingCosts = []
testCosts = []
best_validation_error = 1
trainingAccuracy = 1

for i in range(epochs):

    pYtrain, _ = forward(Xtrain, W1, b1, W2, b2)
    pYtest, _ = forward(Xtest, W1, b1, W2, b2)
    cTrain = cost2(Ytrain.astype(np.int32), pYtrain)
    cTest = cost2(Ytest.astype(np.int32), pYtest)

    trainingCosts.append(cTrain)
    testCosts.append(cTest)

    A2, A1 = forward(X, W1, b1, W2, b2)
    dZ2 = A2 - T
    dW2 = A1.T.dot(dZ2)
    db2 = dZ2.sum(axis=0, keepdims=True)

    # purpose of reg*self.W2 is to make sure the weight is small in order to
    # prevent over fitting
    W2 = W2 - learning_rate * (dW2 + reg * W2)
    b2 = b2 - learning_rate * (db2 + reg * b2)

    dA1 = dZ2.dot(W2.T)
    # 1 - A1 * A1 is derivative of tanh
    dZ1 = dA1 * (1 - A1 * A1)
    dW1 = X.T.dot(dZ1)
    db1 = dZ1.sum(axis=0, keepdims=True)
    W1 = W1 - learning_rate * (dW1 + reg * W1)
    b1 = b1 - learning_rate * (db1 + reg * b1)

    if i % 1000 == 0:
        # testResults is rank one array
        # testResults is nx1 prediction vector containing predicted class
        testResults = np.argmax(pYtest, axis=1)
        # I want testResults to be an nx1 column vector
        testResults = np.reshape(testResults, (testResults.shape[0], 1))
        e = error_rate(Ytest, testResults)
        if e < best_validation_error:
            best_validation_error = e

        testResults = np.argmax(pYtrain, axis=1)
        # I want testResults to be an nx1 column vector
        testResults = np.reshape(testResults, (testResults.shape[0], 1))
        accuracy = error_rate(Ytrain, testResults)
        if accuracy < trainingAccuracy:
            trainingAccuracy = accuracy

        print("i: ", i, "Training cost:", cTrain, "Testing cost:", cTest, "error: ", e, "best_error: ",
              best_validation_error, "accuracy: ", trainingAccuracy)

# after the for loop
print("best_validation_error: ", best_validation_error)
print("Best accuracy: ", trainingAccuracy)
print('model accuracy: ', score(Xtest, Ytest, W1, b1, W2, b2))

legend1, = plt.plot(trainingCosts, label='train_cost')
legend2, = plt.plot(testCosts, label='test cost')
plt.legend([legend1, legend2])
plt.show()







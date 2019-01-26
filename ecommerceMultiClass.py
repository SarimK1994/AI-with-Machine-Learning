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
from ProcessEcommerce import get_data, y2indicator


class ANN(object):
    def __init__(self, M):
        """

        :param M: number of hidden unit in the hidden layer
        """
        self.M = M  # this is the property for the object

    # function for learning
    def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)  # make sure you get random data by using shuffle

        # Make training and validation data set
        Xvalid, Yvalid = X[-100:], Y[-100:]  # Takes last thousand elements and assigns to X and Y valid
        X, Y = X[:-100], Y[:-100]  # Takes all samples up UNTIL last 1000

        N, D = X.shape
        K = len(set(Y[:, 0]))
        T = y2indicator(Y)

        # W1 is weight matrix of hidden layer
        # We want to make sure the variance of W1 is 1/D so that my weight is small
        # Self.W1 makes W1 instance variable for object
        # We initialize W1 randomly using normal distribution

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros((1, self.M))

        # W2 is weight matrix for output layer

        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros((1, K))

        costs = []
        best_validation_error = 1
        # perform epochs number of iterations for gradient descent
        for i in range(epochs):
            # A1 is activation for first hidden layer.
            # A2 is activation for output layer.

            A2, A1 = self.forward(X)
            dZ2 = A2 - T
            dW2 = A1.T.dot(dZ2)
            db2 = dZ2.sum(axis=0, keepdims=True)

            # purpose of reg*self.W2 is to make sure the weight is small in order to
            # prevent over fitting
            self.W2 = self.W2 - learning_rate * (dW2 + reg*self.W2)
            self.b2 = self.b2 - learning_rate * (db2 + reg*self.b2)

            dA1 = dZ2.dot(self.W2.T)
            # 1 - A1 * A1 is derivative of tanh
            dZ1 = dA1 * (1 - A1*A1)
            dW1 = X.T.dot(dZ1)
            db1 = dZ1.sum(axis=0, keepdims=True)
            self.W1 = self.W1 - learning_rate * (dW1 + reg*self.W1)
            self.b1 = self.b1 - learning_rate * (db1 + reg*self.b1)

            # check our cross validation results
            if i % 10 == 0:
                # _ means I do not care about this value
                # A2 is output, A1 is activation

                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid.astype(np.int32), pYvalid)  # cost 2 is used for multi-class classification.
                costs.append(c)
                # testResults is rank one array
                # testResults is nx1 prediction vector containing predicted class
                testResults = np.argmax(pYvalid, axis=1)
                # I want testResults to be an nx1 column vector
                testResults = np.reshape(testResults, (testResults.shape[0], 1))
                e = error_rate(Yvalid, testResults)
                print("i: ", i, "cost: ", c, "error: ", e, "best_error: ", best_validation_error)
                if e < best_validation_error:
                    best_validation_error = e

        # after the for loop
        print("best_validation_error: ", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        """

        :param X: N x D matrix
        :return: N x K matrix normalized matrix (A2) and N x M activation matrix for first
                 hidden layer
        """

        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)  # A1 is the activation value for first hidden layer using tanh function

        #  A2 is the activation for the output layer (prediction value)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = softmax(Z2)  # softmax is used for multi-class classification
        return A2, A1

    def predict(self, X):
        """

        :param X: NxK input matrix
        :return: Nx1 prediction vector
        """
        pY, _ = self.forward(X)
        pred = np.argmax(pY, axis=1)
        pred = np.reshape(pred, (pred.shape[0], 1))
        return pred

    def score(self, X, Y):
        """

        :param X: training set which is 2d array
        :param Y: vector of target
        :return: accuracy of model
        """
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    print("ANN for facial expression")
    X, Y = get_data()  # get the data for all classes
    model = ANN(20)  # 200 hidden unit
    model.fit(X, Y, reg=0, show_fig=True)
    print('model accuracy: ', model.score(X, Y))


if __name__ == '__main__':
    main()

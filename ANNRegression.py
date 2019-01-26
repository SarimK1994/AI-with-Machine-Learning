# Author
# Date
# Purpose: One layer neural network with regression
# Regression means that you try to predict a value such as stock price

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate and plot data
N = 500
# Random number between (-2, +2)
# Uniform distribution
X = np.random.random((N, 2)) * 4 - 2

# Make a saddle shape, which is multiplication between column 0 and column 1
# X[:,0] is all the rows in column 0
Y = X[:, 0]*X[:, 1]
Y = np.reshape(Y, (Y.shape[0], 1))

print("Shape of X", X.shape)
print("Shape of Y", Y.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# Make a neural network and train it
D = 2

# Number of hidden units
M = 100


# Layer 1
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros((1, M))

# Layer 2
W2 = np.random.randn(M, 1) / np.sqrt(M)
b2 = 0


def forward(X):
    Z1 = X.dot(W1) + b1
    A1 = Z1 * (Z1 > 0) # relu activation
    Z2 = A1.dot(W2) + b2
    A2 = Z2 # this is prediction, which is real value

    return A1, A2


# How to train the parameter
def derivative_W2(A1, Y, A2):
    dZ2 = A2 - Y
    dW2 = A1.T.dot(dZ2)
    return dW2


def derivative_b2(Y, A2):
    dZ2 = A2 - Y
    db2 = dZ2.sum(axis=0, keepdims=True)
    return db2


def derivative_W1(X, A1, Y, A2, W2):
    dZ2 = A2 - Y
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * (A1 > 0)  # relu
    dW1 = X.T.dot(dZ1)
    return dW1


def derivative_b1(A1, Y, A2, W2):
    dZ2 = A2 - Y
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * (A1 > 0)  # relu
    db1 = dZ1.sum(axis=0, keepdims=True)
    return db1


def update(X, A1, Y, A2, W1, b1, W2, b2, learning_rate=1e-4):
    gW2 = derivative_W2(A1, Y, A2)
    gb2 = derivative_b2(Y, A2)
    gW1 = derivative_W1(X, A1, Y, A2, W2)
    gb1 = derivative_b1(A1, Y, A2, W2)

    W2 -= learning_rate * gW2
    b2 -= learning_rate * gb1
    W1 -= learning_rate * gW1
    b1 -= learning_rate * gb1

    return W1, b1, W2, b2


# mean square error
def get_cost(Y, A2):
    return((Y - A2) ** 2).mean()


costs = []

for i in range(200):
    A1, A2 = forward(X)
    W1, b1, W2, b2 = update(X, A1, Y, A2, W1, b1, W2, b2)
    cost = get_cost(Y, A2)
    costs.append(cost)
    if i % 25 == 0:
        print(cost)


# plot the cost
plt.plot(costs)
plt.show()

# plot the prediction with data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()



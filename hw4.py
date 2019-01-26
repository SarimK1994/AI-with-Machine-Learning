# Author: Sarim Khan
# Homework for CNN TensorFlow using street numbers data set
# Sample in this data set has 10 digits (0-9)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.io import loadmat
from sklearn.utils import shuffle

# vComments
# convert target vector into N x K indicator matrix
# K is 10, which is the number class.
#


def y2indicator(y):
    N = len(y)
    K = 10
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


# Comments
# Given prediction vector p and target vector t, we will calculate error rate using np.mean
#
def error_rate(p, t):
    return np.mean(t != p)


# Comments
# Implement convpool function
# Parameter of this function include X, which is input image
# W and b is weight and bias of this convolution layer
#
def convpool(X, W, b):
    # X is input image
    # strides=[1,1,1,1] means I move the filter one pixel at a time
    # padding='SAME' means that size of output image is equal to size of input image
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    # image dimensions are expected to be N x width x height x channel
    pool_out = tf.nn.max_pool(
        conv_out,
        ksize=[1, 2, 2, 1],  # pool size
        strides=[1, 2, 2, 1],
        padding='SAME'
    )
    return tf.nn.relu(pool_out)


# implement init_filter function. Parameter is shape of filter and pool size.
# Shape of filter is (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
#
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)

# Rearrange function will rearrange the dimensionality of input matrix X
#


def rearrange(X):
    # input is (32, 32, 3 N)
    # output is (N, 32, 32, 3) For TF, color comes last
    return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)


def main():
    train = loadmat('./large_files/train_32x32.mat') # N = 73257
    test = loadmat('./large_files/test_32x32.mat')  # N = 26032

    # Need to scale! Don't leave as 0..255
    # Y is an N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = rearrange(train['X'])
    # Ytrain is rank-one array
    Ytrain = train['y'].flatten() - 1
    print('Size of Ytrain', len(Ytrain))
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1
    del test
    Ytest_ind = y2indicator(Ytest)

    # Gradient descent parameters
    max_iter = 6
    print_period = 10
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz

    #
    # Get number of Xtrain, which is multiple of batch_sz
    # You could also just do N = N / batch_sz * batch_sz
    #

    Xtrain = Xtrain[:73000]  # remove comma
    Ytrain = Ytrain[:73000]
    Ytrain_ind = Ytrain_ind[:73000]  # THIS IS DIFFERENT FROM ORIGINAL CODE
    Xtest = Xtest[:26000]
    Ytest = Ytest[:26000]
    Ytest_ind = Ytest_ind[:26000]
    print()
    # Xtest.shape(26000, 32, 32, 3)
    # Ytest.shape(26000, )
    print('Xtest.shape', Xtest.shape)
    print('Ytest.shape', Ytest.shape)
    print()

    # Initial weights of fully connected MLP
    # We only have one hidden layer
    # M is the number of units in the hidden layer and K is number of class
    M = 500
    K = 10

    # poolsz is the shape of pool
    poolsz = (2,2)
    #
    # This is the shape of filter for first convpool layer
    # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_shape = (5, 5, 3, 20)

    # Comments
    # call init_filter to get initial value of weight (W1_init)
    #

    W1_init = init_filter(W1_shape, poolsz)

    # Comments
    # Initialize the bias (b1_init) using np.zeros
    # Since we need one bias per each output feature map, the number bias is W1_shape[-1]
    # dtype=np.float(32)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

    # Shape of W2 for second convpool layer
    W2_shape = (5, 5, 20, 50)

    # Comments
    # Similarly initialize W2_init and b2_init
    #

    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # Comments
    # Define initial value of first hidden layer weight (W3_init) using np.random.randn
    # Make sure you normalize the weight using input size and output size of hidden layer
    # Input size of first hidden layer weight is W2_shape[-1]*8*8
    # Output size of first hidden layer weight is M
    W3_init = np.random.randn((W2_shape[-1] * 8 * 8), M) / np.sqrt(W2_shape[-1] * 8 * 8)

    # Why it is 8*8?
    # (First layer) 32x32 16x16 (second layer) 16 x 16 8 x 8
    #

    # Comments
    # Define initial value of first hidden layer bias (b3_init) using np.zeros
    # Size of b3_init is M
    # dtype=np.float32
    b3_init = np.zeros(M, dtype=np.float32)

    # Comments
    # Define weight of output layer (W4_init) and bias (b4_init) output layer
    #
    W4_init = np.random.randn(M, K) / np.sqrt(M)
    b4_init = np.zeros(K, dtype=np.float32)

    # Comments
    # Define X as tf.placeholder. Its data type is tf.float32. Its shape is (None, 32, 32, 3)
    # And it's name is 'X'
    #
    X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='X')

    # Comments
    # Define T as tf.placeholder. Its data type is tf.float32. It's shape is (None, K)
    # And it's name is 'T'
    #
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    # Comments
    # Define tf.Variable W1 and b1 using W1_init.astype(np.float32) and b1_init.astype(np.float32)
    #
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))

    # Comments
    # Similarly, define W2, b2, W3, b3, W4, and b4 as tf.Variable
    #
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    # Comments
    # Define Z1, which is the output of the first convpool layer using convpool function, X, W1, and b1
    #
    Z1 = convpool(X, W1, b1)

    # Comments
    # Define Z2 which is output of second convpool layer using Z1, W2, and b2
    #
    Z2 = convpool(Z1, W2, b2)

    # All of these are in the process of building a graph
    # We need to reshape Z2 for feeding data into network
    # -1 is used to represent 500, which is batch size
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [-1, np.prod(Z2_shape[1:])])

    # Comments
    # Define Z3, which is the output of the hidden layer. Z3 depends on Z2r, W3, and b3
    #

    Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)

    # Comments
    # Define Yish, which is output of the output layer without softmax
    #
    Yish = tf.matmul(Z3, W4) + b4

    # Comments
    # Define cost using tf.reduce_sum and tf.nn.softmax_cross_entropy_with_logits
    #
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

    # Comments
    # Define train_op using train.RMSPropOptimizer
    # Learning_rate = 0.0001
    # Decay = 0.99
    # Momentum = 0.9
    #
    learning_rate = 0.0001
    decay = 0.99
    momentum = 0.9
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum).minimize(cost)

    # Comments
    # Define predict_op using tf.argmax and Yish
    # We'll use this to calculate the error rate
    #
    predict_op = tf.argmax(Yish, 1)

    # **Comments
    # Test our tensorflow code
    # Getting shape of output for each layer using one batch of data
    # Test shape of output for Z1, Z2, Z2r, Z3, and Yish
    print("Z1 Shape:", Z1.shape)
    print("Z2 Shape: ", Z2.shape)
    print("Z2r Shape: ", Z2r.shape)
    print("Z3 Shape: ", Z3.shape)
    print("Yish Shape: ", Yish.shape)
    print()

    t0 = datetime.now()
    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz)]
                Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz)]

                if len(Xbatch) == batch_sz:
                    # Comments
                    # Run train_op using Xbatch and Ybatch for n_batches for max_iter
                    #
                    session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                    if j % print_period == 0:
                        # Due to RAM limitations, we need to have a fixed size input
                        # As a result, we have this ugly total cost and prediction computation
                        # We need to compute the cost and prediction batch by batch. Finally, we add them together
                        test_cost = 0
                        prediction = np.zeros(len(Xtest))

                        # We need to loop through the entire test set and add them all together
                        #

                        for k in range(len(Xtest)):
                            Xtestbatch = Xtest[k * batch_sz:(k * batch_sz + batch_sz)]
                            Ytestbatch = Ytest_ind[k * batch_sz:(k * batch_sz + batch_sz)]
                            test_cost += session.run(cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                            prediction[k * batch_sz:(k * batch_sz + batch_sz)] = session.run(
                                predict_op, feed_dict={X: Xtestbatch})

                            # Comments
                            # Calculate error rate using prediction and Ytest
                            # Print out cost and error for the test set
                            #
                            costs.append(test_cost)
                            err = error_rate(Ytest, prediction)
                            print("test cost:", test_cost, "test errors:", err)

    print("Final Accuracy:", error_rate(Ytest, prediction))

    print("Elapsed time: ", (datetime.now()-t0))
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()

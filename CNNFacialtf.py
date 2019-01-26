# Author
# Date
# Purpose: CNN (Convolutional neural network) Tensor Flow implementation for facial recognition

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from utilfac import getImageData, error_rate, init_weight_and_bias, y2indicator, relu

# Define hidden layer object for regular feed forward network
#


class HiddenLayer(object):
    # M1 is number of unit for input
    # M2 is number of unit for output
    # an_id is id for this hidden layer
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        # W and b is tf.Variable, which can be updated
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]  # parameter for this layer

    # How to perform computation for one layer (computational graph) using function below
    def forward(self, X):
        """

        :param X: Intermediate expression (Tensor Flow object)
        :return: Output expression for hidden layer
        """
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

# init_filter will create initial value of filter using numpy
# Dimension for filter
# filter_width x filter_height x input feature maps(# of input channels) x output feature maps(# of output channels)
# Shape is tuple having 4 members
# Shape is weight matrix for filter, which is 4D
# This will create a 4D matrix of random value


# Input dimension: np.prod(shape[:-1] filter_w * filter*h * input feature map
def init_filters(shape, poolsz):
    # This will produce a 4D filter
    # Then we will normalize the weight
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) +
                                          shape[-1]*np.prod(shape[:-2]/np.prod(poolsz)))
    return w.astype(np.float32)


# Define one convolution and pool layer
class ConvPoolLayer(object):
    # mi is input feature maps
    # mo is output feature maps
    # fw is filter width
    # fh is filter height
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2,2)):
        # (filter_w, filter_h, input feature map, output feature maps)
        sz = (fw, fh, mi, mo)
        # W0 is a 4D matrix
        W0 = init_filters(sz, poolsz)
        self.W = tf.Variable(W0)
        # each filter has one bias. mo is number of filter
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]

    def forward(self, X):
        # X is input image
        # strides=[1, 1, 1, 1] means I move the filter one pixel at a time
        # padding='SAME' means that the size of the output image is equal to the size of input image
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        # Image dimensions are expected to be: N x width x height x channel
        p1, p2 = self.poolsz
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],  # this is the pool size
            strides=[1, p1, p2, 1],
            padding='SAME'
        )
        return tf.nn.relu(pool_out)


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        """

        :param convpool_layer_sizes: List of tuples (num_feature_outmap, filter_w, filter_h)
        :param hidden_layer_sizes: List of hidden layer sizes
        """
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=10e-4, mu=0.99, reg=10e-4,
            decay=0.99999, eps=10e-3, batch_sz=30, epochs=20, show_fig=True):
        # convert everything to float32 so that it can be run in GPU
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y))

        # Make validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)

        print("Shape of X in fit: ", X.shape)
        print("Shape of Y: ", Y.shape)

        Xvalid, Yvalid = X[-1000:], Y[-1000:]  # Validation data set
        X, Y = X[:-1000], Y[:-1000]  # Training data set
        Yvalid_flat = np.argmax(Yvalid, axis=1)

        print("Shape of Yvalid_flat: ", Yvalid_flat.shape)

        # Initialize convpool layers
        N, width, height, c = X.shape  # Number of images, width, height, and channel
        mi = c  # This is your input channel
        # outw and outh are used to calculate the output size of the last conv layer
        outw = width
        outh = height
        # List to hold convolution and pooling layer
        self.convpool_layers = []

        # mo is output channels(number of filter)
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layer_sizes.append(layer)
            outw = outw // 2
            outh = outh // 2  # shrinks the image by a factor of 2
            mi = mo

        # List of hidden layers for fully connected network
        self.hidden_layers = []

        #
        # initialize mlp layers
        # size of input size must be same as output of last convpool layer
        # self.convpool_layer_size[-1][0], this gets the last conv layers channel number
        # M1 is the number of neurons for input of first hidden layer
        M1 = self.convpool_layer_sizes[-1][0] * outw * outh  # M1 is last channel * outw * outh

        count = 0
        # create each hidden layer object
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # define last hidden layer to output (which is logistic layer)
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')

        # Collect params for later use
        self.params = [self.W, self.b]

        for h in self.convpool_layer:
            self.params += h.params

        for h in self.hidden_layers:
            self.params += h.params

        # Set up TensorFlow functions and variables
        # Placeholders for input data
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')

        # This will form the computational graph
        act = self.forward(tfX)  # act means output of final layer
        # total cost for all weights and bias in all layers
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])

        # In order to calculate cost; you need act (activation) and tfY
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                label='tfY'
            )
        ) + rcost

        prediction = self.predict(tfX)

        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

        n_batches = N // batch_sz

        # Test each convolution layer and pooling layer for one batch to make sure it works
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            Xbatch_T = X[2 * batch_sz: (2 * batch_sz + batch_sz)]
            Ybatch_T = Y[2 * batch_sz: (2 * batch_sz + batch_sz)]

            convpool_layer1 = self.convpool_layer[0]
            conv1 = tf.nn.conv2d(tfX, convpool_layer1.W, strides=[1, 1, 1, 1], padding='SAME')
            conv1_bias = tf.nn.bias_add(conv1, convpool_layer1.b)
            p1, p2 = convpool_layer1.poolsz
            pool_out1 = tf.nn.max_pool(
                conv1_bias,
                ksize=[1, p1, p2, 1],
                strides=[1, p1, p2, 1],
                padding='SAME'
            )
            conv1_finalOutput = tf.nn.relu(pool_out1)

            conv1_shape = session.run(tf.shape(conv1), feed_dict={tfX: Xbatch_T})
            print("conv1_shape: ", conv1_shape)
            print()

            conv1_final_shape = session.run(tf.shape(conv1_finalOutput), feed_dict={tfX: Xbatch_T})
            print("conv1 final shape: ", conv1_final_shape)
            print()

            shape_finaloutputs = session.run(tf.shape(act), feed_dict={tfX: Xbatch_T})
            print("Shape of final output", shape_finaloutputs)

            value_cost = session.run(cost, feed_dict={tfX: Xbatch_T, tfY: Ybatch_T})
            print("cost before training", value_cost)

        costs = []
        init = tf.global_variables_initializer()

        with tf.Session as session:
            session.run(init)

            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid}) # Gives cost
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})  # Gives prediction
                        e = error_rate(Yvalid_flat, p)
                        print("iteration: ", i, j, "cost: ", c, "error rate:", e)

            if show_fig:
                plt.plot(costs)
                plt.show()



    def forward(self, X):
        Z = X

        for c in self.convpool_layer:
            Z = c.forward(Z)

        # get the shape of the output of the last conv layer and turn into a list
        Z_shape = Z.get_shape().as_list()

        # Dimension of output of conv layer
        # N x width x height x numOfChannels
        # -1 can be used to infer the shape since we do not know how many samples we used
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])

        for h in self.hidden_layers:
            Z = h.forward(Z)

        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)



def main():
    # Get 4D image data
    X, Y = getImageData()
    # Reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))
    print('X.shape in main: ', X.shape)

    model = CNN(
        # 20 means # of filters of output channel
        # 5 x 5 is filter size
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )

    model.fit(X, Y)


if __name__ == '__main__':
    main()
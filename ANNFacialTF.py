# Author
# Date
# Purpose: ANN Tensorflow version for facial recognition

# All variables in TensorFlow need to be of the same type in order to run this in GPU

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utilfac import getData, y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle


# define HiddenLayer class
class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        """

        :param M1: This is the input size of the hidden layer
        :param M2: This is the output size of the hidden layer
        :param an_id: This is a tensorflow variable id
        """
        W, b = init_weight_and_bias(M1, M2)
        # I initialize the self.W and self.b using numpy array
        # self.W and self.b are tensorflow variables

        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        """

        :param X: Intermediate tensorflow expression
        :return: output expression for hidden layer
        """
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

        # mu is used for momentum
        # decay is used for RMSprop
        # reg is used for regularization
    def fit(self, X, Y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3,
            epochs=400, batch_sz=100, show_fig=False):

        print("facial expression recognition")
        print("shape of X", X.shape)
        print("shape of Y", Y.shape)
        print("sample of X", X[:4])
        print("sample of Y which is label", Y[:4])
        print()

        K = len(set(Y))
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        print("K: ", K)
        print("Y which is indicator matrix", Y[:4])

        # Make a validation set
        Xvalid, Yvalid = X[-1000:], Y[-1000:]

        # Yvalid_flat is rank-one array containing label
        # turn the indicator matrix to label matrix
        Yvalid_flat = np.argmax(Yvalid, axis=1)
        print()
        print("shape of Yvalid", Yvalid)
        print("shape of Yvalid_flat", Yvalid_flat.shape)
        print()

        # make a new X, Y training set
        X, Y = X[:-1000], Y[:-1000]

        print("shape of X after final manipulation", X.shape)
        print("shape of Y after final manipulation", Y.shape)
        print("shape of Xvalid after final manipulation", Xvalid.shape)
        print("shape of Yvalid after final manipulation", Yvalid.shape)

        # N is number of samples
        # D is number of features
        N, D = X.shape

        self.hidden_layers = []
        # Initially, input is D, which is dimensionality of input data
        M1 = D
        count = 0
        # create list containing hidden layer
        # output of previous layer becomes input of next layer
        for M2 in self.hidden_layer_sizes:
            # M1 is number of input neuron, M2 is number of output neuron
            # count gives us identifier for each layer
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            count += 1

        # output layer includes K output units
        # self.W is weight between last hidden layer and output layer
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))


        # collect parameters for later use
        self.params = [self.W, self.b]

        # self.hidden_layers is the list of hidden layers
        for h in self.hidden_layers:
            self.params += h.params

        # define the computational graph for tensorflow
        # set up input variable for tensorflow
        # None means that I can pass any shape for batch size.
        # tfX is used for training data
        # tfT is indicator matrix
        tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')
        # act produces computational graph
        act = self.forward(tfX)  # act means activation in this case

        # calculate l2_loss
        # rcost is regularization of weight
        # self.params is weight and bias for all layers
        rcost = reg * sum([tf.nn.l2_loss(p) for p in self.params])

        # calculate the total cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits==act,
            labels=tfT
        )) + rcost

        # get the prediction given input variable
        prediction = self.predict(tfX)

        # define train_op, which is the tensor flow optimization functionality
        # tensorflow will calculate the derivative automatically
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

        n_batches = N // batch_sz

        costs = []

        # initialize all variables
        init = tf.global_variables_initializer()

        # test each function in tensorflow
        with tf.Session() as session:
            session.run(init)
            batch_no = 2
            Xbatch_T = X[batch_no*batch_sz:(batch_no*batch_sz+batch_sz)]
            Ybatch_T = Y[batch_no*batch_sz:(batch_no*batch_sz+batch_sz)]

            # test forward function
            hiddenLayer1 = self.hidden_layers[0]
            hiddenLayer2 = self.hidden_layers[1]
            hiddenLayer3 = self.hidden_layers[2]
            A1 = hiddenLayer1.forward(tfX)
            A2 = hiddenLayer2.forward(A1)
            A3 = hiddenLayer3.forward(A2)

            shape_A3 = session.run(tf.shape(A3), feed_dict={tfX: Xbatch_T})
            print("shape of A3", shape_A3)
            print()

            shape_act = session.run(tf.shape(act), feed_dict={tfX: Xbatch_T})
            print("shape of act: ", shape_act)

            for i in range(100):
                # running RMSprop for 100 iterations
                session.run(train_op, feed_dict={tfX: Xbatch_T, tfT: Ybatch_T})
                value_cost = session.run(cost, feed_dict={tfX: Xbatch_T, tfT: Ybatch_T})
                print("value of cost after update", value_cost)
                print()

        # Loop through and call train_op
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    # Xbatch is training data in one batch
                    # Ybatch is target indicator matrix in one batch
                    Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                    Ybatch = Y[j * batch_sz:(j*batch_sz + batch_sz)]
                    # train neural network for one iteration
                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                    # print cost and error on validation set at every 60 steps
                    if j % 60 == 0:
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(c)
                        # produce prediction for validation set
                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        print("i:", i, "j:", j, "cost:", c, "error_rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    # forward function
    # given input X and produce output
    # this function produces tensorflow computational graph
    def forward(self, X):
        Z = X  # set Z to be current input
        for h in self.hidden_layers:
            Z = h.forward(Z)
            # I am going through hidden layer to do computation
            # in tensorflow, we only return activation not softmax
        return tf.matmul(Z, self.W) + self.b  # final output

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, 1) # return label array


def main():
    X, Y = getData()
    # This is neural network with 3 hidden layers
    model = ANN([2000, 1000, 500])
    model.fit(X, Y, show_fig=True)


if __name__ == '__main__':
    main()

#Author:
#Date
#Purpose: Use logistic regression for facial expression

import numpy as np   # np is the short name for numpy cause we be lazy
import matplotlib.pyplot as plt #same still lazy af
from sklearn.utils import shuffle
from LogisticFacialUtil import getBinaryData, sigmoid, sigmoid_cost, error_rate, cost2

#We will code our logistic regression model while we're in class so we can use it as and object
#This is very similar to how machine learning models work in sklearn
#

class LogisticModel(object):
    #This is object initializatioin nothing happens with init
    def __init__(self):
        pass # nothing happens when pass is executed

    # fit function is used to train the model
    # fit is the learning function
    def fit(self, X, Y, learning_rate =10e-7, reg=0*10e-22, epochs=100000, show_fig = False):
        #The more documentation in python the better or else you will get confused ~ Zhong 2018
        """

        :param X: input data which is 2d numpy array
        :param Y: 1d numpy target vector (is it happy sad retarded?)
        :param learning_rate: how fast I will change my weight parameter
        :param reg: regularization to prevent overfitting
        :param epochs: how many iterations I will run my gradient descent
        :param show_fig: show figure of cost curve
        :return:
        """

        #Load the shape of the data "very important" x.shape and y.shape
        print("shape of X at the begining of fix", X.shape)
        print("shape of Y at the begining of fix", Y.shape)

        # produce training and validation set
        # training set is used to train the model
        # validation set is used to validate performance of the data
        # shuffle make sure all data is randomly sorted

        X,Y = shuffle(X,Y)
        #X and Y is a 2d array
        # Below the negative signs move backwards from the end of the array
        # Last 1000 samples are validation dataset (xValid and yValid), whatever is left is the training set
        Xvalid, Yvalid = X[-1000:],Y[-1000:]
        X,Y = X[:-1000],Y[:-1000]

        print("shape of X after dividing", X.shape)
        print("shape of Y after dividing", Y.shape)
        print("shape of Xvalid after dividing", Xvalid.shape)
        print("shape of Yvalid after dividing", Yvalid.shape)

        #Here you watches that the shape of xvalid and yvalid are 1000 he also checks the original dataset
        #has decrease in dimention accordingly constantly checking the shape

        #N is the number of samples(8876), D is the dimension of the array(2304)
        N,D = X.shape

        #define weight vector W and bias b
        #here we divide by np.sqrt(D) to make sure our weight the sqrt is an arbitrary function to make it small
        #if the weight is too big your gradient will explode
        #we make sure variance of wweight is 1/D
        #self.W can W the instance variable of the class
        self.W = np.random.randn(D,1) / np.sqrt(D)
        self.b = 0

        print("Shape of W", self.W.shape)

        costs = []
        best_validation_error = 1

        #perform multiple iteration of gradient descent
        # epochs number of iteration
        # essence of the learning
        for i in range(epochs):

            #pY is vector of prediction probability (N x 1)
            pY = self.forward(X)

            # gradient descent step
            # dZ is n x 1 vector
            # pY is n x 1
            # - is element-wise operation
            # dZ change in z with respect to L in theory dz=(dL/dZ) back in theory (math notes)
            # reg*self.W is regularization term to prevent wieght becoming too big
            dZ = pY - Y
            # one step of gradient descent
            # every iteration change W a little bit until reaching the global minimum
            self.W = self.W - learning_rate*(X.T.dot(dZ) + reg*self.W)
            self.b = self.b - learning_rate*(dZ.sum() + reg*self.b)


            #error is calculated from validation set Not training set
            # print out info every 20 iterations
            if i % 20 == 0:
                # prediction vector from validation set
                pYvalid = self.forward(Xvalid)

                # calculate error for validation data-set
                # if you look at the source code of sigma cost it is basically our formula from theory (math notes)
                # cost is loss here, numpy code to calculate loss how we get our loss
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c) #append c to costs list
                # calculating the error rate compared with the target
                e = error_rate(Yvalid, np.round(pYvalid))
                print('i: ',i, 'costs: ', c, ' error:', e)
                if  e < best_validation_error:
                    best_validation_error = e

        #outside the loop
        print("best validation error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
            """

            :param self:
            :param X: 2d array
            :return: N x 1 vector of prediction (Double)
            """

            # sigmoid normalizes the data
            # dot product is the .dot() and W is the weights
            return sigmoid(X.dot(self.W) + self.b)



#Main function outside of class
def main():
    print(np.version)
    print()

    #This thing below returns binary data fer2013 source code
    X, Y = getBinaryData()
    #New shape of data after calling the above function
    print('shape of X before class imbalance main', X.shape)
    print('shape of Y before class imbalance main', Y.shape)


    N1,D1 = X.shape
    #figure out what is probability that y = 0
    p0 = np.sum(Y == 0) / float(N1)
    #np.sum adds allllll the numbers together then
    p1 = np.sum(Y == 1) / float(N1)
    print("p0 before balance:", p0, " p1 before balance: ", p1)

    #Number of sample for class 0 is 9 times more than class 1
    #This is highly imbalanced dataset. The logistic regression does not work well will highly imbalanced data

    #The following code below balances the data set, it is not necessarily part of the engine but instead
    #this code manipulates data
    # get sample for class 0
    X0 = X[Y[:, 0] == 0]  # changed : is required
    # get the sample for class 1
    X1 = X[Y[:, 0] == 1]  # changed
    # copy class1 for 9 times
    X1 = np.repeat(X1, 9, axis=0)  # class 1 is 9 times less than other classes
    # combine class 0 and class 1 row-wise
    X = np.vstack([X0, X1])  # row-wise operation
    # create new targe vector
    Y = np.array([0] * len(X0) + [1] * len(X1))
    Y = np.reshape(Y, (Y.shape[0], 1))

    print("shape of X after imbalance", X.shape)
    print("shape of Y after imbalance", Y.shape)
    # Always with AI code you must keep track of shape, shape is very important information to assure not mutation

    N,D = X.shape #shape after balancing data
    p0 = np.sum(Y == 0) / float(N)
    p1 = np.sum(Y == 1) / float(N)
    print("p0 after balance: ", p0, "p1 after balance: ", p1)

    model = LogisticModel() #create the model

    # build the logistic model
    model.fit(X, Y, show_fig = True)


if __name__ == '__main__':
    main()
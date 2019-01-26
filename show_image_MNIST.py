# This is used to show MNIST dataset
# 4/15/2018
# show images
#

import numpy as np
import matplotlib.pyplot as plt

from utilMNIST import get_normalized_data


def main():
    # X is (35887, 2304)
    # Y is (35887, 1)
    X, Y = get_normalized_data()

    print("shape of X", X.shape)
    print("shape of Y", Y.shape)

    print('I get data')

    while True:
        # rotate between 10 class
        #
        for i in range(10):
            x, y = X[Y[:,0]==i], Y[Y[:,0]==i]   # pick the sample from one class
            # shape of x for class 1  (547, 2304)
            # shape of y(547, 1)
            print("class: ", i)
            print("shape of x", x.shape)
            print("shape of y", y.shape)
            N = len(y)
            j = np.random.choice(N)   # randomly pick one sample

            # because the data currently live in the flat vectors,
            # we have to reshape it to 48 x 48 image
            # cmap means color map
            #
            plt.imshow(x[j].reshape(28,28), cmap='gray')
            plt.title(i)
            plt.show()

        prompt = input('Quit? Enter Y:\n')

        if prompt == 'Y':
            break

if __name__ == '__main__':
    main()

# 4/14/2018
# show images
#
# how to get data
# Learn facial expressions from an image
# Challenges in Representation Learning: Facial Expression Recognition Challenge
# Kaggle

import numpy as np
import matplotlib.pyplot as plt

from LogisticFacialUtil import getData

# class is from 0 to 6
#
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    # X is (35887, 2304)
    # Y is (35887, 1)
    X, Y = getData(balance_ones=True)

    print("shape of X", X.shape)
    print("shape of Y", Y.shape)

    print('I get data')

    while True:
        # rotate between 7 class
        #
        for i in range(7):
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
            plt.imshow(x[j].reshape(48,48), cmap='gray')
            plt.title(label_map[y[j,0]])
            plt.show()

        prompt = input('Quit? Enter Y:\n')

        if prompt == 'Y':
            break

if __name__ == '__main__':
    main()

# At first, we started with implementation of neural network with numpy and python
# Then we moved to TensorFlow, which is higher level of deep learning library
# Now, we will move to an even higher level, which is Keras. Under the hood, Keras will use TensorFlow library

# Type of model we will use:
from keras.models import Sequential

# Different layers we will use
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Helper method
# Keras requires that target is one-hot encoding
#


def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


# Load the data
data = pd.read_csv("./large_files/fashion-mnist_train.csv")
data = data.as_matrix()  # Turn into numpy array
np.random.shuffle(data)

print("Shape of data", data.shape)
print()

# Data needs to be shaped of N x height x width x color
# The image is 28 x 28 in gray scale
# We need to normalize the input
# X will be from column 1 onwards
# Y is on column 0
X = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)

print("Shape of X", X.shape)
print("Shape of Y", Y.shape)
print()

# Get the number of class K
K = len(set(Y))

# By default, Keras wants one-hot encoded tables
Y = y2indicator(Y)

model = Sequential()

# Let us define 3 convolutional layers
model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3)))

# BatchNormalization is used to normalize output for each layer
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

# Second conv-pool layer
# I define 64 3x3 filters below
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

# Third conv-pool layer
# I define 128 3x3 filters
model.add(Conv2d(filters=128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

# Flatten my last conv-pool layer
model.add(Flatten())

# Add 1 dense hidden layer with 300 hidden neurons
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0, 2))  # Randomly drop the node for regularization

# Finally I define the output layer
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Summary of Model")
print(model.summary())
print()

r = model.fit(X, Y, validation_split=0.33, epochs=10, batch_size=32)
print("Returned: ", r)
print((r.history.keys()))


# Loss for training and cross-validation
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')  # Validation accuracy
plt.legend()
plt.show()






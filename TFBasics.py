# Author
# Date
# Purpose: practice tensorflow basics

import numpy as np
import tensorflow as tf

# In tensorflow, tf.placeHolder is used to feed actual training examples
# tf.Variable is used for trainable variables such as weights and bias

# Let us create 5 x 5 matrix
# You need data type for each element and its type
A = tf.placeholder(tf.float32, shape=(5,5), name='A')

# But shape and name are optional
# We can create a vector without giving a shape or name
v = tf.placeholder(tf.float32)

# matmul is matrix multiplication
# right now I am defining the computational graph. There is no data so far
w = tf.matmul(A, v)

# In Tensorflow, you do the actual work in a session
with tf.Session() as session:
    # I want to figure out, what is the shape of w?
    # feed_dict tells what A is and what v is
    # tf.shape(w) is the output
    shape_w = session.run(tf.shape(w), feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})
    print()
    print("output type of shape_w", type(shape_w))
    print("shape of w", shape_w)
    print()

    # w is computation for this run
    output_w = session.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})

    print()
    print("output type of output_w", type(output_w))
    print("output_w", output_w)
    print()

    shape_v = session.run(tf.shape(v), feed_dict={v: np.random.randn(5,2)})
    print("shape of v", shape_v)
    print()

    # In short, tf.Variable is used for trainable variables such as weights or bias
    # tf.placeholder is used to feed actual training examples
    #

# A tf.Variable can be initialized with numpy array or a tf array, or more correctly,
# Anything that can be turned into a tf tensor
#

shape = (2,2)
# You use numpy array to initialize tf.variable
y = tf.Variable(np.random.randn(2, 2))

# You can use tf array for initialization
x = tf.Variable(tf.random_normal(shape))

# You can also make your variable a scalar
t = tf.Variable(0)

# You need to initialize the variable first
init = tf.global_variables_initializer()

with tf.Session() as session:
    # And then, "run" the init operation
    out = session.run(init)

    # You can print out the value of tensor flow variable using eval() function
    #
    result = x.eval()
    print()
    print('value of x', result)
    print('type of x', type(result))
    print('shape of x', result.shape)
    print()
    result = t.eval()
    print('value of t', result)
    print('type of t', type(result))
    print()
# Let us now try to find the minimum of a single cost function
u = tf.Variable(20.0)
cost = u*u + u + 1.0  # cost can be considered as variable
# 0.3 is the learning rate.
# for minimize, you tell it which expression you want to minimize.
# You do not worry about how to find out derivative and update your u
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
init = tf.global_variables_initializer()
# let us run a session again
with tf.Session() as session:
    session.run(init)

    for i in range(12):
        # for each iteration, you do one update of your weight u
        # weight update is automated, but loop itself its not.
        session.run(train_op)
        print('i=', i, 'cost:', cost.eval(), 'u:', u.eval())
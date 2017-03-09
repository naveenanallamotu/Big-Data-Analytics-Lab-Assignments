
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt

rnd = np.random


data = pandas.read_csv("cars.csv", header=0,delimiter=',')
rnd_indices = 200
traindata = np.array(list(data["speed"][:rnd_indices]))
traindata2 = np.array(list(data["dist"][:rnd_indices]))

testdata = np.array(list(data["speed"][rnd_indices:]))
testdata2 = np.array(list(data["dist"][rnd_indices:]))


Xx = tf.placeholder("float")

Yy = tf.placeholder("float")

# create a shared variable for the weight matrix
weigh = tf.Variable(rnd.randn(), name="weights")
ba = tf.Variable(rnd.randn(), name="bias")

# prediction function
y_model = tf.add(tf.multiply(Xx, weigh), ba)

# squared error
cost = tf.reduce_sum(tf.square(y_model - Yy)) / (2 * traindata.shape[0])

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializing the variables
init = tf.global_variables_initializer()
# you need to initialize variables
sess.run(init)
print(sess.run(weigh), sess.run(ba))

for i in range(1000):
    for(x,y) in zip(traindata, traindata2):
        sess.run(train_op, {Xx:x, Yy: y})

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={Xx: traindata, Yy: traindata2})

print("Training cost=", training_cost, "W=", sess.run(weigh), "b=", sess.run(ba), '\n')
testing_cost = sess.run(
    tf.reduce_sum(tf.square(y_model - Yy)),
    feed_dict={Xx: testdata, Yy: testdata2})
print("Testing cost=", testing_cost)
print("Absolute square loss difference:", abs(
    training_cost - testing_cost))

plt.plot(traindata, traindata2, 'ro', label='Original data')
plt.plot(traindata, sess.run(weigh) * traindata + sess.run(ba), label='Fitted line')
plt.plot(testdata, testdata2, 'ro', c='g', label='Test data')
plt.plot(testdata, sess.run(weigh) * testdata + sess.run(ba), label='Fitted line')
plt.legend()
plt.show()
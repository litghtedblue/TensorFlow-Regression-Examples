#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets


def init_weights(shape):
#    return tf.Variable(tf.random_normal(shape, stddev=0.33))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.33))

def init_b(i):
    return tf.Variable(tf.constant(1.0,shape=[i]))


def model(x, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_in, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    h = tf.nn.relu(tf.matmul(x, w_h)+b_h)
    h = tf.nn.dropout(h, p_keep_in)

    h2 = tf.nn.relu(tf.matmul(h, w_h2)+b_h2)
    h2 = tf.nn.dropout(h2, p_keep_in)

    output = tf.nn.relu(tf.matmul(h2, w_o)+b_o)
    output = tf.nn.dropout(output, p_keep_hidden)
    return output

def loss(model, y):
    return tf.reduce_mean(tf.square(model - y), name="loss")

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

diabetes = datasets.load_diabetes()
data = diabetes["data"].astype(np.float32)
target = diabetes['target'].astype(np.float32).reshape(len(diabetes['target']), 1)


MAX_SIZE = data.shape[0]
TEST_N = 100
N = MAX_SIZE - TEST_N
BATCH_SIZE = 10
MAX_STEPS = 1000

train_x, test_x = np.vsplit(data, [N])
train_y, test_y = np.vsplit(target, [N])

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)



x = tf.placeholder("float", [None, 10])
y = tf.placeholder("float", [None, 1])

w_h = init_weights([10, 128])
w_h2 = init_weights([128, 256])
w_o = init_weights([256, 1])

b_h= init_b(128)
b_h2= init_b(256)
b_o= init_b(1)


#p_keep_in = tf.placeholder("float")
p_keep_in = tf.placeholder(tf.float32)
#p_keep_hidden = tf.placeholder("float")
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(x, w_h,b_h,w_h2,b_h2,w_o,b_o, p_keep_in, p_keep_hidden)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

loss_value=loss(py_x,y)
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss_value)
#predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
#with tf.Session() as sess:
    # you need to initialize all variables
#    tf.global_variables_initializer().run()

#    for i in range(100):
#        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
#            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
#                                          p_keep_in: 0.8, p_keep_hidden: 0.5})
#        print(i, np.mean(np.argmax(teY, axis=1) ==
#                         sess.run(predict_op, feed_dict={X: teX, 
#                                                         p_keep_in: 1.0,
#                                                         p_keep_hidden: 1.0})))


best = float("inf")
init_op = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init_op)
    for step in range(MAX_STEPS + 1):
        for i in range(int(N / BATCH_SIZE)):
            batch = BATCH_SIZE * i
            train_batch_x = train_x[batch:batch + BATCH_SIZE]
            train_batch_y = train_y[batch:batch + BATCH_SIZE]

            loss_train = sess.run(loss_value, feed_dict={
                                  x: train_batch_x, y: train_batch_y,
                                  p_keep_in: 1.0, p_keep_hidden: 1.0})
            sess.run(train_op, feed_dict={
                     x: train_batch_x, y: train_batch_y,
                     p_keep_in: 0.8, p_keep_hidden: 0.5})

        if loss_train < best:
            best = loss_train
            best_match = sess.run(py_x, feed_dict={
                x: test_x, y: test_y, p_keep_in: 1.0, p_keep_hidden: 1.0})

        if step % 10 == 0:
            cor = np.corrcoef(best_match.flatten(), test_y.flatten())
            print('step : {}, train loss : {}, test cor : {}'.format(
                step, best, cor[0][1]))



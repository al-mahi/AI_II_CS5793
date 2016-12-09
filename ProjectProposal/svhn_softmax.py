#!/usr/bin/python

from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def svhn(file_name, one_hot=True):
    """
     :rtype x:np.ndarray
     :rtype y:np.ndarray
    """
    train = loadmat(file_name=file_name)
    x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    y = train['y'].flatten()
    np.place(y, y==10, 0)
    x = np.array(x.reshape(x.shape[0], 32 * 32 * 3))

    if one_hot:
        z = np.zeros(shape=(y.shape[0], 10))
        z[range(y.shape[0]), y] = 1
        y = z

    return x, y


def softmax_sklearn():
    X, Y = svhn('train_32x32.mat', one_hot=False)
    f, t = svhn('test_32x32.mat', one_hot=False)

    X = np.array(X.reshape(X.shape[0], 32 * 32 * 3))
    f = np.array(f.reshape(f.shape[0], 32 * 32 * 3))
    clf = SGDClassifier(loss='log')
    clf.fit(X, Y)
    y_hat = clf.predict(f)
    print("sklearn {:.3%}".format(accuracy_score(t, y_hat)))


def softmax_tf():
    X, Y = svhn('train_32x32.mat')
    f, t = svhn('test_32x32.mat')

    X = np.array(X.reshape(X.shape[0], 32 * 32 * 3))
    f = np.array(f.reshape(f.shape[0], 32 * 32 * 3))

    x = tf.placeholder(tf.float32, [None, 32*32*3])
    W = tf.Variable(tf.zeros([32*32*3, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    batch = 100

    for i in range(1000):
        if (i+1)*batch > X.shape[0]:
            break
        batch_xs, batch_ys = (X[i*batch: (i+1)*batch], Y[i*batch: (i+1)*batch])
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("tf {:.3%}".format(sess.run(accuracy, feed_dict={x: f, y_: t})))


if __name__ == "__main__":
    softmax_sklearn()
    softmax_tf()


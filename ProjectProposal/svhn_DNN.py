#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import os

def svhn(file_name, one_hot=True):
    """
     :rtype x:np.ndarray
     :rtype y:np.ndarray
    """
    train = loadmat(file_name=file_name)
    x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    y = train['y'].flatten()
    np.place(y, y==10, 0)
    x = np.mean(x, axis=3)
    x = np.array(x.reshape(x.shape[0], 32 * 32))

    if one_hot:
        z = np.zeros(shape=(y.shape[0], 10))
        z[range(y.shape[0]), y] = 1
        y = z

    return x, y

def CNN():

    traindata, trainlabels = svhn('train_32x32.mat')
    testdata, testlabels = svhn("test_32x32.mat")
    batch_size = 100
    N = traindata.shape[0]
    N_test = testdata.shape[0]
    learning_rate = 1e-4

    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 32*32])
    W = tf.Variable(tf.zeros([32*32, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_true = tf.placeholder(tf.float32, [None, 10])
    keepratio = tf.placeholder(tf.float32)

    z = tf.matmul(x, W) + b
    tf.summary.histogram('pre_activations', z)

    y_est = tf.nn.softmax(z)
    error = 0.5*tf.reduce_sum(tf.square(y_true - y_est))
    tf.scalar_summary("cost", error)
    train_step = tf.train.AdamOptimizer().minimize(error)



    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)

    merge_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('summary_{}'.format(learning_rate), graph_def=sess.graph_def)

    print('\nDeep learning:')

    n_output = 10

    weights = {
      'wc1': tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1)),
      'wc2': tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1)),
      'wf1': tf.Variable(tf.truncated_normal([8*8*64, 1024], stddev=0.1)),
      'wf2': tf.Variable(tf.truncated_normal([1024, n_output], stddev=0.1))
    }
    biases = {
      'bc1': tf.Variable(tf.random_normal([32], stddev=0.1)),
      'bc2': tf.Variable(tf.random_normal([64], stddev=0.1)),
      'bf1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
      'bf2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }

    cnn = network(x, weights, biases, keepratio)
    y_conv = cnn['out']

    # tf.summary.image('input', tf.reshape(x, [-1, 32, 32, 1]), max_outputs=5)
    tf.summary.image("try1", tf.reshape(cnn['conv1'], [-1, 32, 32, 1]), max_outputs=5)
    # Note: Don't add a softmax reducer in the network if you are going to use this
    # cross-entropy function

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_true))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary("acc", accuracy)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)

    last_ckpt= 0
    if False:
        save_path = "opt_param/model_{}_{}.ckpt".format(learning_rate, last_ckpt)
        saver.restore(sess=sess, save_path=save_path)

    for i in range(last_ckpt+1, 400):
        idxs = np.random.randint(low=0, high=N, size=batch_size)
        if i % 10 == 0:
            save_path = saver.save(sess, "opt_param/model_{}_{}.ckpt".format(learning_rate, i))
            print("Model saved in file: %s" % save_path)

            train_accuracy = sess.run(accuracy, feed_dict={
                x: traindata[idxs], y_true: trainlabels[idxs], keepratio: 1.0})

            print("step %d, training accuracy %g" % (i, train_accuracy))

            summary_str = sess.run(merge_summary, feed_dict={x: traindata[idxs], y_true: trainlabels[idxs]})
            summary_writer.add_summary(summary_str, i)

        sess.run(train_step, feed_dict={x: traindata[idxs], y_true: trainlabels[idxs], keepratio: 0.5})

    idxs = np.random.randint(low=0, high=N_test, size=N_test/2)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: testdata[idxs], y_true: testlabels[idxs], keepratio: 1.0}))


def network(input, w, b, keepratio):
    # INPUT
    input_r = tf.reshape(input, shape=[-1, 32, 32, 1])
    # CONV LAYER 1
    conv1 = tf.nn.conv2d(input_r, w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b['bc1']))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # CONV LAYER 2
    conv2 = tf.nn.conv2d(pool1, w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b['bc2']))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # VECTORIZE
    dense1 = tf.reshape(pool2, [-1, w['wf1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    fc1 = tf.nn.relu(tf.add(tf.matmul(dense1, w['wf1']), b['bf1']))
    fc_dr1 = tf.nn.dropout(fc1, keepratio)
    # FULLY CONNECTED LAYER 2
    out = tf.add(tf.matmul(fc_dr1, w['wf2']), b['bf2'])
    # RETURN
    retval = { 'input_r': input_r, 'conv1': conv1, 'pool1': pool1,
               'conv2': conv2, 'pool2': pool2, 'dense1': dense1,
               'fc1': fc1, 'fc_dr1': fc_dr1, 'out': out
    }
    return retval


if __name__=="__main__":
    CNN()
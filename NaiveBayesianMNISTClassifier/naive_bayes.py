#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 2: Naive Bayesian classifiers
Solution for Part 1
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def mnist_data(fname):
    """
    Read data from MNIST file
    :param fname: MNIST file path
    :return: MNIST data
    :rtype: np.ndarray
    """
    with open(fname, 'rb') as training_images_file:
        training_images = training_images_file.read()
        training_images = bytearray(training_images)
        training_images = training_images[16:]
    training_images = np.array(training_images, dtype="float64") > (255./2.)
    data_size = training_images.shape[0]
    image_size = 28 * 28
    num_of_img = data_size / image_size
    # figure, axes = plt.subplots(nrows=10, ncols=10)
    # ind = 1
    # print(training_images.shape, image_size, num_of_img)
    # for axis in axes.flat:
    #     axis.imshow(training_images[(ind-1) * image_size: ind * image_size].reshape(28, 28), cmap='bone')
    #     axis.set_axis_off()
    #     ind += 1
    # plt.show()
    return training_images.reshape(num_of_img, image_size)


def mnist_labels(fname):
    """
    Read labels from MNIST file
    :param fname: MNIST file path
    :return: MNIST labels in shape (N, 1)
    :rtype: np.ndarray
    """
    with open(fname, 'rb') as training_label_file:
        training_labels = training_label_file.read()
        training_labels = bytearray(training_labels)
        training_labels = training_labels[8:]
    training_labels = np.array(training_labels)
    num_of_labels = training_labels.shape[0]
    return training_labels.reshape(num_of_labels, 1)


if __name__ == "__main__":
    x = mnist_data('train-images.idx3-ubyte')
    y = mnist_labels('train-labels.idx1-ubyte')
    k = 10
    n, d = x.shape
    counts = np.bincount(y.flatten())
    prior = counts / float(n)
    theta = np.zeros(shape=(k, d))

    for dig in range(k):
        mask = (y.flatten() == dig)
        #  remember the prior formula is (n + 1)/(N+k) so adding 1 in nominator
        white_pixel_count = np.sum(x[mask], axis=0, dtype="float64") + 1.
        #  remember the prior formula is (n + 1)/(N+k) so adding k=10 in denominator
        theta[dig] += white_pixel_count / (counts[dig] + k)

    log_complement = np.log(1. - theta)

    x = mnist_data('t10k-images.idx3-ubyte')
    y = mnist_labels('t10k-labels.idx1-ubyte')

    log_theta = np.log(theta)

    y_hat = np.zeros(y.shape)
    log_prior = np.log(prior)
    for i in range(x.shape[0]):
        log_likelyhood = np.sum(log_theta[:, x[i].flatten()], axis=1) + np.sum(log_complement[:, np.logical_not(x[i].flatten())], axis=1)
        log_posterior = log_prior + log_likelyhood
        y_hat[i] = np.argmax(log_posterior)

    print("MNIST data classification accuracy using Naive Bayes {:.2%}".format(float(sum(y == y_hat)) / y.shape[0]))




















#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 2: Naive Bayesian classifiers
Solution for Part 2
"""

from __future__ import print_function
import numpy as np
import scipy.stats
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
    training_images = np.array(training_images, dtype="float64")
    data_size = training_images.shape[0]
    image_size = 28 * 28
    num_of_img = data_size / image_size
    # figure, axes = plt.subplots(nrows=2, ncols=2)
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


def pdf_gaussian(x, mu, sigma):
    return 1./((np.sqrt(2.*np.pi) * sigma) * np.exp((x - mu)**2./(2.*sigma**2.)))

if __name__ == "__main__":
    x = mnist_data('train-images.idx3-ubyte')
    y = mnist_labels('train-labels.idx1-ubyte')
    k = 2
    n = 5
    fives = x[(y == 5).flatten()]
    others = x[np.logical_not((y == 5).flatten())]
    print(fives.shape)
    print(others.shape)
    x = np.vstack((
        fives[:n],
        others[:n]
    ))
    y = np.vstack((
        np.ones((n, 1)),
        np.zeros((n, 1))
    ))

    n, d = x.shape
    mu = [0.5, 0.5]
    # k = 0 means is fives and k=1 means not 5
    # 2*d is for mean and variance of each pixel

    theta = np.vstack((
        zip(np.mean(x[:n], axis=0, dtype="float64"), np.std(x[:n], axis=0, dtype="float64")),
        zip(np.mean(x[n:], axis=0, dtype="float64"), np.std(x[n:], axis=0, dtype="float64"))
    ))
    print(theta.shape)
    print(theta[0, :2])
    print(theta[1, :2])

    x = mnist_data('t10k-images.idx3-ubyte')
    y = mnist_labels('t10k-labels.idx1-ubyte')

    # because log of 0 is infinite imposing epsilon 0.00000000001 as minimum small value
    epsilon = 0.00000000001
    log_theta = np.log(theta.clip(min=epsilon))



























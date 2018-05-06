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


if __name__ == "__main__":
    x = mnist_data('train-images.idx3-ubyte')
    y = mnist_labels('train-labels.idx1-ubyte')

    mask = np.random.random(size=(x.shape[0])) < .9
    x_train = x[mask.flatten()]
    y_train = y[mask.flatten()]
    x_test = x[np.logical_not(mask.flatten())]
    y_test = y[np.logical_not(mask.flatten())]

    k = 2
    n = 1000
    fives = x_train[(y_train == 5).flatten()]
    others = x_train[np.logical_not((y_train == 5).flatten())]

    x = np.vstack((
        fives[:n],
        others[:n]
    ))
    y = np.vstack((
        np.ones((n, 1)),
        np.zeros((n, 1))
    ))

    d = x.shape[1]

    # k = 0 means is fives and not five otherwise
    # 2*d is for mean and variance of each pixel
    mu = np.zeros(shape=(k, d))

    mu[0] = np.mean(x[:n], axis=0, dtype="float64")
    mu[1] = np.mean(x[n:], axis=0, dtype="float64")
    std_0 = np.std(x[:n], dtype="float64")
    std_1 = np.std(x[n:], dtype="float64")

    x = x_test
    y = y_test

    mask = (y == 5)
    y[mask] = 1
    y[np.logical_not(mask)] = 0

    y_hat = np.zeros(y.shape)
    log_likelyhood_ratio = np.zeros(y.shape)
    # log loss ratio of type I and type II errors follow lec slide 4
    threshold = np.log(np.array([5., 2., 1., 0.5, .2]))

    plot_x = []
    plot_y = []

    for i in range(x.shape[0]):
        gaussians_for_fives = scipy.stats.norm(mu[0, :], std_0)
        gaussians_for_other = scipy.stats.norm(mu[1, :], std_1)
        likelyhood_five = np.array(gaussians_for_fives.pdf(x[i].flatten()))
        likelyhood_other = np.array(gaussians_for_other.pdf(x[i].flatten()))
        log_likelyhood_five = np.sum(np.log(likelyhood_five))
        log_likelyhood_other = np.sum(np.log(likelyhood_other))
        log_likelyhood_ratio[i] = log_likelyhood_five - log_likelyhood_other

    for tau in threshold:
        mask = (log_likelyhood_ratio > tau)
        y_hat[mask] = 1
        y_hat[np.logical_not(mask)] = 0

        true_pos_mask = np.logical_and(y.flatten() == 1, y_hat.flatten() == 1)
        true_neg_mask = np.logical_and(y.flatten() == 0, y_hat.flatten() == 0)
        false_pos_mask = np.logical_and(y.flatten() == 0, y_hat.flatten() == 1)
        false_neg_mask = np.logical_and(y.flatten() == 1, y_hat.flatten() == 0)

        FPR = float(sum(false_pos_mask)) / (sum(true_neg_mask) + sum(false_pos_mask))
        TPR = float(sum(true_pos_mask)) / (sum(true_pos_mask) + sum(false_neg_mask))

        plot_x.append(FPR)
        plot_y.append(TPR)

    plt.plot(plot_x, plot_y, 'r-')
    plt.title("ROC Diagram")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


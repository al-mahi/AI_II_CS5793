#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 1: Basic classifiers
Solution for Part 5
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from KDTree import cKDTree

if __name__ == "__main__":
    """
    Part#5 Increasing complexity
    Create a new set of training and test data. This time, each classification will be produced by multiple
    distributions, rather than just one. Draw 1000 samples each from ten different overlapping Gaussian
    distributions. Five of them should be labeled as class 0 and the others class 1. An example is shown
    in Figure 2. Perform the same linear and nearest neighbor classification processes, calculate the
    classification accuracy and plot the results.
    """
    N = 1000
    mean0 = [0, 0]
    cov0 = [[.5, 0], [0, 20]]
    X0 = np.random.multivariate_normal(mean0, cov0, N)

    mean1 = [2.5, 7]
    cov1 = [[3, 0], [0, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, N)

    mean2 = [2.5, -6]
    cov2 = [[3, 0], [0, 1]]
    X2 = np.random.multivariate_normal(mean2, cov2, N)

    mean3 = [5, 0.5]
    cov3 = [[0.5, 0], [0, 10]]
    X3 = np.random.multivariate_normal(mean3, cov3, N)

    mean4 = [0, -4]
    cov4 = [[10, 0], [2, 1]]
    X4 = np.random.multivariate_normal(mean4, cov4, N)

    # class 2
    mean5 = [3, 3]
    cov5 = [[2, 0], [0, 2]]
    X5 = np.random.multivariate_normal(mean5, cov5, N)

    mean6 = [3.5, 2]
    cov6 = [[3, 2], [0, 1]]
    X6 = np.random.multivariate_normal(mean6, cov6, N)

    mean7 = [4, 1.5]
    cov7 = [[1, 0], [0, 5]]
    X7 = np.random.multivariate_normal(mean7, cov7, N)

    mean8 = [1.5, 1]
    cov8 = [[2, 3], [0, 10]]
    X8 = np.random.multivariate_normal(mean8, cov8, N)

    mean9 = [5, -2]
    cov9 = [[10, 3], [0, 1]]
    X9 = np.random.multivariate_normal(mean9, cov9, N)

    X = np.vstack((X0,
                   X1,
                   X2,
                   X3,
                   X4,
                   X5,
                   X6,
                   X7,
                   X8,
                   X9))
    y = np.vstack((np.zeros(shape=(5*N, 1), dtype='int'),
                   np.ones(shape=(5*N, 1), dtype='int')))

    mask = np.random.random(size=(10*N)) < .8

    training_X = X[mask]
    training_y = y[mask]
    test_X = X[np.logical_not(mask)]
    test_y = y[np.logical_not(mask)]

    plt.plot(X[:5 * N, 0], X[:5 * N, 1], 'x', c='b', label="data calss 0")
    plt.plot(X[5 * N:, 0], X[5 * N:, 1], 'x', c='r', label="data calss 1")
    plt.title("Data from multiple multi variant 2d Gaussian")
    plt.legend(loc="lower right")
    plt.show()

    beta = np.linalg.inv(training_X.T.dot(training_X)).dot(training_X.T).dot(training_y)
    y_hat = test_X.dot(beta) >= 0.5  # differs from problem description. (?)

    print("Linear Classifier accuracy from multiple multi variant 2d Gaussian distribution: {:.2%}".format(float(sum(y_hat == test_y)) / len(test_y)))

    training_from_class0 = training_X[training_y.flatten() == 0]
    training_from_class1 = training_X[training_y.flatten() == 1]
    correct_from_class0 = test_X[np.logical_and(test_y.flatten() == 0, y_hat.flatten() == 0)]
    correct_from_class1 = test_X[np.logical_and(test_y.flatten() == 1, y_hat.flatten() == 1)]
    incorrect_from_class0 = test_X[np.logical_and(test_y.flatten() == 0, y_hat.flatten() == 1)]
    incorrect_from_class1 = test_X[np.logical_and(test_y.flatten() == 1, y_hat.flatten() == 0)]

    plt.plot(training_from_class0[:, 0], training_from_class0[:, 1], 'x', c='b', label='Training set from class 0')
    plt.plot(training_from_class1[:, 0], training_from_class1[:, 1], 'x', c='r', label='Training set from class 1')
    plt.plot(correct_from_class0[:, 0], correct_from_class0[:, 1], 'o', c='y', label='Correctly classified test set from class 0')
    plt.plot(correct_from_class1[:, 0], correct_from_class1[:, 1], 's', c='c', label='Correctly classified test set from class 1')
    plt.plot(incorrect_from_class0[:, 0], incorrect_from_class0[:, 1], '.', c='m', label='Incorrectly classified test set from class 0')
    plt.plot(incorrect_from_class1[:, 0], incorrect_from_class1[:, 1], '.', c='k', label='Incorrectly classified test set from class 1')
    plt.legend(loc='lower right', fontsize='small')
    plt.title("Linear Classifier from multiple multi variant 2d Gaussian distribution")
    plt.show()

    kdtreeClassifier = cKDTree(training_X)
    y_hat = training_y[kdtreeClassifier.query(test_X[:], k=1)[1]]

    print("KDTree Classifier accuracy from multiple multi variant 2d Gaussian distribution: {:.2%}".format(
        float(sum(y_hat == test_y)) / len(test_y)))

    correct_from_class0 = test_X[np.logical_and(test_y.flatten() == 0, y_hat.flatten() == 0)]
    correct_from_class1 = test_X[np.logical_and(test_y.flatten() == 1, y_hat.flatten() == 1)]
    incorrect_from_class0 = test_X[np.logical_and(test_y.flatten() == 0, y_hat.flatten() == 1)]
    incorrect_from_class1 = test_X[np.logical_and(test_y.flatten() == 1, y_hat.flatten() == 0)]

    plt.plot(training_from_class0[:, 0], training_from_class0[:, 1], 'x', c='b', label='Training set from class 0')
    plt.plot(training_from_class1[:, 0], training_from_class1[:, 1], 'x', c='r', label='Training set from class 1')
    plt.plot(correct_from_class0[:, 0], correct_from_class0[:, 1], 'o', c='y',
             label='Correctly classified test set from class 0')
    plt.plot(correct_from_class1[:, 0], correct_from_class1[:, 1], 's', c='c',
             label='Correctly classified test set from class 1')
    plt.plot(incorrect_from_class0[:, 0], incorrect_from_class0[:, 1], '.', c='m',
             label='Incorrectly classified test set from class 0')
    plt.plot(incorrect_from_class1[:, 0], incorrect_from_class1[:, 1], '.', c='k',
             label='Incorrectly classified test set from class 1')
    plt.legend(loc='lower right', fontsize='small')
    plt.title("KDtree Classifier from multiple multi variant 2d Gaussian distribution")
    plt.show()





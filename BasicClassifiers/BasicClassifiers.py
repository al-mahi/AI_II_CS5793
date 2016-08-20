#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 1: Basic classifiers
Solution for Part 2,3,4,5
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from KDTree import cKDTree

if __name__ == "__main__":

    """
    Part#2 Constructing a simple data set
    Draw 5000 data points from each of two different 2D multivariate Gaussian distributions, with
    means and covariances of your own choosing. The two distributions should be distinct but over-
    lapping, as illustrated in Figure 1. You may use np.random.randn to generate your x1 and x2
    coordinates independently, but this will limit your distributions to those with diagonal covariance
    matrices. Alternately, you may use np.random.multivariate normal to sample from an arbitrarily-
    covarying 2D distribution. You should end up with a matrix X, which will be 10000 X 2 in size,
    and a column vector y of 10000 elements. The ith element of y should be 0 if the ith row of X
    came from your first distribution, and 1 if it came from the second.
    Randomly partition X and y into training and test sets. The easiest way to do this is to
    construct a 10000-element column vector of booleans and use it as a mask.
    """

    N = 5000
    mean0 = [2, 2]
    cov0 = [[1, 0], [1, 2]]  # diagonal covaria
    mean1 = [4, 4]
    cov1 = [[3, 0], [0, 1]]  # diagonal covaria

    X = np.vstack((np.random.multivariate_normal(mean0, cov0, N),
                   np.random.multivariate_normal(mean1, cov1, N)))
    y = np.vstack((np.zeros(shape=(N, 1), dtype='int'),
                   np.ones(shape=(N, 1), dtype='int')))

    mask = np.random.random(size=(2*N)) < .8

    training_X = X[mask]
    training_y = y[mask]
    test_X = X[np.logical_not(mask)]
    test_y = y[np.logical_not(mask)]

    plt.plot(X[:N, 0], X[:N, 1], 'x', c='b', label="data calss 0")
    plt.plot(X[N:, 0], X[N:, 1], 'x', c='r', label="data calss 1")
    plt.title("Data from multi variant 2d Gaussian")
    plt.legend(loc="lower right")
    plt.show()

    """
    Part#3 Linear classifier
    Using your training set (which the equation below calls X), construct the maximum-likelihood
    linear least squares function beta to fit your data.
    Plot
    - Training set elements from class 0
    - Training set elements from class 1
    - Correctly classified test set elements from class 0
    - Correctly classified test set elements from class 1
    - Incorrectly classified test set elements from class 0
    - Incorrectly classified test set elements from class 1
    """
    beta = np.linalg.inv(training_X.T.dot(training_X)).dot(training_X.T).dot(training_y)
    y_hat = test_X.dot(beta) >= 0.5

    print("Linear Classifier accuracy: {:.2%}".format(float(sum(y_hat == test_y)) / len(test_y)))

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
    plt.title("Linear Classifier")
    plt.show()

    """
    Part#4 Nearest neighbors classification
    Using the same training and test sets as before, construct a kD tree from your training set. Classify
    each element of your test set by looking up its nearest neighbor from the training set and assigning
    yy_hat to be whatever value y belongs to the nearest training example.
    As before, calculate and show the classification accuracy and produce a similar plot.
    """

    kdtreeClassifier = cKDTree(training_X)
    y_hat = training_y[kdtreeClassifier.query(test_X[:], k=1)[1]]

    print("KDTree Classifier accuracy: {:.2%}".format(float(sum(y_hat == test_y)) / len(test_y)))

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
    plt.title("KDtree Classifier")
    plt.show()


































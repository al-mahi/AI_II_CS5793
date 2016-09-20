#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 2: More Nearest Neighbour
Solution for Part 3
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
    training_images = np.array(training_images, dtype="float64")
    data_size = training_images.shape[0]
    image_size = 28 * 28
    num_of_img = data_size / image_size
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
    training_labels = np.array(training_labels, dtype="int64")
    num_of_labels = training_labels.shape[0]
    return training_labels.reshape(num_of_labels, 1)


if __name__ == "__main__":
    """
    Using the original (non-thresholded, 0-255 scale) MNIST training set, cut it down significantly to
    make the computational complexity of a high-dimensional nearest-neighbors search a little more
    tractable. Create a training set consisting of 200 examples of each of three classes, the numerals
    1, 2 and 7. Don't bother with a KD tree remember that in high-dimensional spaces, a KD tree
    is no better than brute force search. This is why we are keeping the size down to 600. Go ahead
    and implement a brute force nearest neighbors search, computing the 784-dimensional Euclidean
    distance
    Conduct a 5-fold cross validation on your training set, testing the NN classifier on five candidate
    classifier models, those with K = {1, 3, 5, 7, 9}. Determine the best candidate model.

    """
    x = mnist_data('train-images.idx3-ubyte')
    y = mnist_labels('train-labels.idx1-ubyte')

    ones = x[(y == 1).flatten()]
    twos = x[(y == 2).flatten()]
    sevens = x[(y == 7).flatten()]

    n = 200
    x = np.vstack((
        ones[:n],
        twos[:n],
        sevens[:n]
    ))
    y = np.vstack((
        np.zeros((n, 1), dtype="int64") + 1,
        np.zeros((n, 1), dtype="int64") + 2,
        np.zeros((n, 1), dtype="int64") + 7
    ))

    # shuffle
    ind = np.arange(3*n)
    np.random.shuffle(ind)
    x = x[ind.flatten()]
    y = y[ind.flatten()]

    Ks = np.array([1, 3, 5, 7, 9])
    avg_accuracies = np.zeros(Ks.shape[0])

    for k in range(Ks.shape[0]):
        num_of_folds = 5
        fold_size = x.shape[0] / num_of_folds

        accuracies = np.zeros(num_of_folds)
        for i in range(num_of_folds):
            mask = np.ones(shape=x.shape[0], dtype="bool")
            mask[i*fold_size:(i+1)*fold_size] = False
            train_x = x[mask.flatten()]
            train_y = y[mask.flatten()]
            validation_x = x[np.logical_not(mask.flatten())]
            validation_y = y[np.logical_not(mask.flatten())]
            validation_y_hat = np.zeros(validation_y.shape)
            for j in range(validation_x.shape[0]):
                def dist(a, b):
                    return np.array([np.sqrt(np.sum((a - bi) ** 2.)) for bi in b])

                ind = np.argsort(dist(validation_x[j], train_x))
                ind = ind[:Ks[k]]
                votes = np.zeros(10)
                for ii in ind:
                    votes[train_y[ii]] += 1
                validation_y_hat[j] = np.argmax(votes)
            accuracies[i] = float(sum(validation_y_hat == validation_y)) / validation_y_hat.shape[0]
        avg_accuracies[k] = float(sum(accuracies)) / num_of_folds
        print("{}-NN accuracy {} avg_accuracy {:.2%}".format(Ks[k], accuracies, avg_accuracies[k]))

    best_k = Ks[np.argmax(avg_accuracies)]
    print("K={}-NN has best average accuracy on training set.".format(best_k))

    """
    Create a test data set from the MNIST test data, consisting of 50 examples of each of the three
    classes we are considering. Using the NN model selected by your model validation, classify your
    test set. Compare the accuracy of your model on your test set with its cross-validation accuracy.
    """

    m = 50
    test_x = mnist_data('t10k-images.idx3-ubyte')
    test_y = mnist_labels('t10k-labels.idx1-ubyte')

    ones = test_x[(test_y == 1).flatten()]
    twos = test_x[(test_y == 2).flatten()]
    sevens = test_x[(test_y == 7).flatten()]

    test_x = np.vstack((
        ones[:m],
        twos[:m],
        sevens[:m]
    ))

    test_y = np.vstack((
        np.zeros((m, 1), dtype="int64") + 1,
        np.zeros((m, 1), dtype="int64") + 2,
        np.zeros((m, 1), dtype="int64") + 7
    ))

    # shuffle
    ind = np.arange(3*m)
    np.random.shuffle(ind)
    test_x = test_x[ind.flatten()]
    test_y = test_y[ind.flatten()]

    accuracy_best_k = 0.
    test_y_hat = np.zeros(test_y.shape)
    for j in range(test_x.shape[0]):
        def dist(a, b):
            return np.array([np.sqrt(np.sum((a - bi) ** 2.)) for bi in b])

        ind = np.argsort(dist(test_x[j], x))
        ind = ind[:best_k]
        votes = np.zeros(10)
        for ii in ind:
            votes[y[ii]] += 1
        test_y_hat[j] = np.argmax(votes)
    accuracy_best_k = float(sum(test_y_hat == test_y)) / test_y_hat.shape[0]
    print("{}-NN accuracy on test set is {:.2%}".format(best_k, accuracy_best_k))

    """
    For each of the three classes, plot some test set examples which your classifier identified correctly
    and some for which it failed. Can you identify any patterns which seem likely to lead to failure?
    """

    def organize_data(img_data):
        if img_data.shape[0] == 0 or img_data is None:
            return None
        img_data = img_data[:10]
        harr = []
        for pi in range(img_data.shape[0]):
            harr.append(img_data[pi].reshape(28, 28))
        return np.hstack(harr)

    plt.set_cmap("bone")
    correctly_classified_ones = test_x[np.logical_and((test_y_hat == 1).flatten(), (test_y == 1).flatten())]
    plt_321 = organize_data(correctly_classified_ones)
    if plt_321 is not None:
        ax = plt.subplot(321)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_321)
        plt.title("correctly_classified_as_ones")

    incorrectly_classified_ones = test_x[np.logical_and((test_y_hat == 1).flatten(), (test_y != 1).flatten())]
    plt_322 = organize_data(incorrectly_classified_ones)
    if plt_322 is not None:
        ax = plt.subplot(322)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_322)
        plt.title("incorrectly_classified_as_ones")

    correctly_classified_twos = test_x[np.logical_and((test_y_hat == 2).flatten(), (test_y == 2).flatten())]
    plt_323 = organize_data(correctly_classified_twos)
    if plt_323 is not None:
        ax = plt.subplot(323)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_323)
        plt.title("correctly_classified_as_twos")

    incorrectly_classified_twos = test_x[np.logical_and((test_y_hat == 2).flatten(), (test_y != 2).flatten())]
    plt_324 = organize_data(incorrectly_classified_twos)
    if plt_324 is not None:
        ax = plt.subplot(324)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_324)
        plt.title("incorrectly_classified_as_twos")

    correctly_classified_sevens = test_x[np.logical_and((test_y_hat == 7).flatten(), (test_y == 7).flatten())]
    plt_325 = organize_data(correctly_classified_sevens)
    if plt_325 is not None:
        ax = plt.subplot(325)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_325)
        plt.title("correctly_classified_as_sevens")

    incorrectly_classified_sevens = test_x[np.logical_and((test_y_hat == 7).flatten(), (test_y != 7).flatten())]
    plt_326 = organize_data(incorrectly_classified_sevens)
    if plt_326 is not None:
        ax = plt.subplot(326)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(plt_326)
        plt.title("incorrectly_classified_as_sevens")

    plt.suptitle("It seems like the tails of the figuresfooled the classifier; "
                 "if the lower tail of 2 is shortened then it has been classified as seven, whereas the if the upper tail\n"
                 "of 7 is shortened it is classified as 1", fontsize=16)
    plt.show()








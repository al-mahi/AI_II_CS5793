#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 4: Part 2 K Means
"""

from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
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


def init_mu(x, y, K):
    N = x.shape[0]
    M = x[0].shape[0]
    mu = np.zeros(shape=(K, M))
    for i in range(K):
        mu[i] = x[np.random.randint(low=0, high=N)]
    done = [False for i in range(K)]
    ind = np.random.randint(low=0, high=N, size=N)
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    for i in range(N):
        min_d = np.inf
        min_k = -1
        for k in range(K):
            if np.linalg.norm(mu[k] - x[i]) < min_d:
                min_d = np.linalg.norm(mu[k] - x[i])
                min_k = k
        if min_k == -1: continue
        if not done[min_k]:
            mu[min_k] = x[i]
            done[min_k] = True
        if sum(done) == K:
            break
    return mu


def init_cheat_mu(x, y, K):
    N = x.shape[0]
    M = x[0].shape[0]
    mu = np.zeros(shape=(K, M))
    taken = np.zeros(N, dtype='bool')

    ind = np.random.randint(low=0, high=N, size=N)
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    for i in range(K):
        for j in range(y.shape[0]):
            if i == y[j] and not taken[j]:
                mu[i] = x[j]
                taken[j] = True
                break
    return mu


def k_means(K, x, mu):
    """
    :type x: np.ndarray
    :type mu: np.ndarray
    :rtype Z: np.ndarray
    :rtype Z: np.ndarray
    """
    print("K Means:")
    N = x.shape[0]
    limit = 100
    Z = np.zeros(shape=(N, K))
    for t in range(limit):
        changed = False
        for i in range(N):
            min_d = np.inf
            min_k = 0
            for k in range(K):
                d = np.linalg.norm(mu[k] - x[i])
                if d < min_d:
                    min_d = d
                    min_k = k
            if Z[i, min_k] != 1.:
                Z[i, :] = 0.
                Z[i, min_k] = 1.
                changed = True

        if not changed:
            break

        cost = 0.
        for k in range(K):
            ind = np.where(Z[:, k] == 1.)[0]
            # slow code because of vstack
            # cluster = x[ind[0]]
            # for i in range(1, len(ind)):
            #     cluster = np.vstack((cluster, x[ind[i]]))
            cluster = np.zeros(x[0].shape)
            for i in range(len(ind)):
                cluster += x[ind[i]]
            mu[k] = cluster / float(len(ind))

            for i in range(cluster.shape[0]):
                cost += np.linalg.norm(mu[k]-cluster[i])
        print("#{} cost={}".format(t, cost))

    return Z, mu


if __name__ == "__main__":
    x = mnist_data('t10k-images.idx3-ubyte')
    y = mnist_labels('t10k-labels.idx1-ubyte')
    y_hat = np.zeros(y.shape)
    N = x.shape[0]
    M = x[0].shape[0]
    K = 10
    Z = np.zeros(shape=(N, K))

    # init mu
    mu1 = x[np.random.randint(low=0, high=N, size=K)]
    mu2 = init_mu(x, y, K)
    mu3 = init_cheat_mu(x, y, K)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mu1[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid for randomly chosen $\mu$")
    plt.savefig("kmeans1.png", format='png')
    plt.show()

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mu2[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid chosen by K-means++")
    plt.savefig("kmeans2.png", format='png')
    plt.show()

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(mu3[ind-1, :].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("centroid chosen by seeing label")
    plt.savefig("kmeans3.png", format='png')
    plt.show()



    Z1, m1 =k_means(K, x, mu1)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m1[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid for randomly chosen $\mu$")
    plt.savefig("kmeans4.png", format='png')
    plt.show()

    Z2, m2 =k_means(K, x, mu2)
    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m2[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid chosen by K-means++")
    plt.savefig("kmeans5.png", format='png')
    plt.show()

    Z3, m3 =k_means(K, x, mu3)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(m3[ind-1].reshape(28, 28), cmap='bone')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid chosen by seeing label")
    plt.savefig("kmeans6.png", format='png')
    plt.show()

    Z4, m4 =k_means(3, x, mu2)
    ind0 = np.asarray(np.where(Z4[:, 0] == 1), dtype='int')[0]
    ind1 = np.asarray(np.where(Z4[:, 1] == 1), dtype='int')[0]
    ind2 = np.asarray(np.where(Z4[:, 2] == 1), dtype='int')[0]

    np.random.shuffle(ind0)
    np.random.shuffle(ind1)
    np.random.shuffle(ind2)

    figure, axes = plt.subplots(nrows=3, ncols=10)
    axes.flat[0].imshow(m4[0].reshape(28, 28), cmap='bone')
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])
    axes.flat[0].set_title("Means")
    axes.flat[0].set_ylabel("k=0", rotation='horizontal')
    for i in range(1, 10, 1):
        axes.flat[i].imshow(x[ind0[i]].reshape(28, 28), cmap='bone')
        axes.flat[i].set_axis_off()
    axes.flat[10].imshow(m4[1].reshape(28, 28), cmap='bone')
    axes.flat[10].set_axis_off()
    axes.flat[10].set_title("k=1", loc='left')
    for i in range(11, 20, 1):
        axes.flat[i].imshow(x[ind1[i-11]].reshape(28, 28), cmap='bone')
        axes.flat[i].set_axis_off()
    axes.flat[20].imshow(m4[2].reshape(28, 28), cmap='bone')
    axes.flat[20].set_axis_off()
    axes.flat[20].set_title("k=2")
    for i in range(21, 30, 1):
        axes.flat[i].imshow(x[ind2[i-21]].reshape(28, 28), cmap='bone')
        axes.flat[i].set_axis_off()
    plt.suptitle("centroid chosen by seting K = 3")
    plt.savefig("kmeans7.png", format='png')
    plt.show()









#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 4: Part 2 K Means
"""

from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

if __name__ == "__main__":
    # x, y = svhn('train_32x32.mat', one_hot=False)
    x, y = svhn('test_32x32.mat', one_hot=False)
    np.place(y, y==10, 0)
    y_hat = np.zeros(y.shape)
    N = x.shape[0]
    M = x[0].shape[0]
    K = 10
    mu3 = init_cheat_mu(x, y, K)

    kmeans2 = KMeans(n_clusters=K, random_state=0, init=mu3).fit(x)
    kmeans1 = KMeans(n_clusters=K, random_state=0, init='k-means++').fit(x)
    # print(x.shape, N, M, x[:100].shape, kmeans1.labels_.shape, kmeans1.cluster_centers_.shape, kmeans1.cluster_centers_.shape )

    acc1 = np.sum(kmeans1.labels_ == y, dtype='float') / float(N)
    acc2 = np.sum(kmeans2.labels_ == y, dtype='float') / float(N)

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(kmeans1.cluster_centers_[ind-1].reshape(32, 32, 3))
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid init by K-means++\naccuracy {:.3%}".format(acc1))
    plt.savefig("kmeans1.png", format='png')
    plt.show()

    figure, axes = plt.subplots(nrows=5, ncols=2)
    ind = 1
    for axis in axes.flat:
        axis.imshow(kmeans2.cluster_centers_[ind-1].reshape(32, 32, 3), cmap='hot')
        axis.set_axis_off()
        ind += 1
    plt.suptitle("K-means centroid init by Selection\naccuracy {:.3%}".format(acc2))
    plt.savefig("kmeans2.png", format='png')
    plt.show()













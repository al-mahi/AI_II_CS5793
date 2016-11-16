#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 3: Regressions
Solution for Part 4 Linear Reg Bayesian MAP
"""
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize


def flower_to_float(s):
    d = {
        "Iris-setosa": 0.,
        'Iris-versicolor': 1.,
        'Iris-virginica': 2.
    }
    return d[s]


def PHI(x):
    """
    :type x: np.ndarray
    :rtype: np.ndarray
    """
    m = x.shape[0]
    mu = np.linspace(x.min(), x.max(), m)
    sigma = np.fabs(mu[0] - mu[1])
    return np.exp(-(x - mu)**2 / (2.*sigma**2))


def f(w, x, t, alpha):
    """
    Bayesian MAP.
    :param w: weights of radial basis function
    :type w: np.ndarray
    :param x: features
    :type x: np.ndarray
    :param t: labels
    :type t: np.ndarray
    :param alpha: precision param
    :return: optimized weights
    """
    neg_log_prior = np.array((alpha/2.) * w.T.dot(w))
    N = x.shape[0]
    K = 3
    neg_log_likeyhood= 0.
    for n in range(N):
        cost = 0.
        for k in range(K):
            print(w[k:k+5].T)
            print(x[n])
            print(PHI(x[n]))
            cost += t[n, k] * w[k:k+5].T.dot(PHI(x[n]))
        for l in range(K):
            cost -= np.log(w[l:l+5].T.dot(PHI(x[n])))
        neg_log_likeyhood += cost
    return neg_log_prior - neg_log_likeyhood


if __name__ == "__main__":
    irises = np.loadtxt('iris.data', delimiter=',', converters={4: flower_to_float})
    N = irises.shape[0]
    K = 3
    x = np.array(irises[:, :4])
    x = np.hstack((np.ones(shape=(N, 1)), x))
    y = np.zeros(shape=(N, K))
    # y[n, i] == 1 iff nth data is from class i
    for i in range(N):
        y[i, int(irises[i, 4])] = 1

    # shuffle
    ind = np.arange(N)
    np.random.shuffle(ind)
    data_x = np.array(x[ind.flatten()])
    data_y = np.array(y[ind.flatten()])

    x = data_x[:N/2]
    t = data_y[:N/2]
    x_test = data_x[N/2:]
    t_test = data_y[N/2:]

    alpha = 0.003126
    w_init = np.ones(15)

    # as minimization takes some time save the result and reuse
    if True:
        w_hat = minimize(f, w_init, args=(x, t, alpha)).x
        np.save("opt_w_MAP_regression", w_hat)
    else:
        w_hat = np.load("opt_w_MAP_regression.npy")

    t_hat = []
    for i in range(x_test.shape[0]):
        z = np.zeros(K)
        for k in range(K):
            z[k] = w_hat[k:k+5].T.dot(PHI(x_test[i]))
        s = z / np.sum(z)
        t_hat.append(s.argmax())
    y = np.array([c.argmax() for c in t_hat])

    print("overall accuracy = {:.3%}".format(float(sum(t_hat == y)) / len(y)))


#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 3: Regressions
Solution for Part 3 Linear Reg Baysian MAP
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def PHI(x, m=0):
    """
    :type x: np.ndarray
    :param basis: polynomial or radial
    :rtype: np.ndarray
    """
    return np.power(x, range(m))

if __name__=="__main__":
    beta = .0025
    data = np.loadtxt("crash.txt")
    training = data[:-1:2]
    test = data[1::2]
    N = training.shape[0]
    x = np.array(training[:, 0]).reshape(N, 1)
    t = np.array(training[:, 1]).reshape(N, 1)
    x_test = np.array(test[:, 0]).reshape(N, 1)
    t_test = np.array(test[:, 1]).reshape(N, 1)

    opt_rms = np.inf
    opt_w = None

    test_opt_rms = np.inf
    test_opt_alpha = None

    RMS = []
    RMS_test = []
    L = 50

    for alpha in np.logspace(-8, 0, 100):
        phi = PHI(x, L)
        w = np.linalg.solve(phi.T.dot(phi) + (alpha/beta) * np.identity(n=L), phi.T.dot(t))
        # E = .5 * np.linalg.norm(t - phi.dot(w))**2
        # rms = np.sqrt(2. * E / N)
        # RMS.append(rms)
        # if rms < opt_rms:
        #     opt_rms = rms
        #     opt_m = L
        #     opt_w = w

        phi = PHI(x_test, L)
        E_test = .5 * np.linalg.norm(t_test - phi.dot(w))**2
        rms = np.sqrt(2. * E_test / N)
        RMS_test.append(np.sqrt(2. * E_test / N))
        if rms < test_opt_rms:
            test_opt_rms = rms
            test_opt_w = w
            test_opt_alpha = alpha

    xmin = min(np.min(x), np.min(x_test))
    xmax = max(np.max(x), np.max(x_test))
    ymin = min(np.min(t), np.min(t_test))
    ymax = max(np.max(t), np.max(t_test))
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    NN = data.shape[0]
    xx = np.array(data[:, 0]).reshape(NN, 1)
    tt = np.array(data[:, 1]).reshape(NN, 1)

    nx = np.linspace(start=0., stop=60., num=1000).reshape(1000, 1)
    phi = PHI(nx, m=L)
    ny = phi.dot(test_opt_w)

    plt.plot(xx, tt, label="data")
    plt.plot(nx, ny, label="plot logspace")
    plt.xlabel("time")
    plt.ylabel("acceleration")
    plt.title("best model on test on data alpha={:.6f}".format(test_opt_alpha))
    plt.legend()
    plt.show()


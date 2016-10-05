#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 3: Regressions
Solution for Part 1 Linear Reg Radial Basis
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def PHI(x, m=0):
    """
    :type x: np.ndarray
    :rtype: np.ndarray
    """
    mu = np.linspace(x.min(), x.max(), m)
    sigma = np.fabs(mu[0] - mu[1])
    return np.exp(-(x - mu)**2 / 2.*sigma**2)

if __name__=="__main__":
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
    opt_m = 0

    test_opt_rms = np.inf
    test_opt_w = None
    test_opt_m = 0

    RMS = []
    RMS_test = []

    for L in range(5, 30, 5):
        phi = PHI(x, L)
        w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(t))
        E = .5 * np.linalg.norm(t - phi.dot(w))**2
        rms = np.sqrt(2. * E / N)
        RMS.append(rms)
        if rms < opt_rms:
            opt_rms = rms
            opt_m = L
            opt_w = w

        phi = PHI(x_test, L)
        E_test = .5 * np.linalg.norm(t_test - phi.dot(w))**2
        rms = np.sqrt(2. * E_test / N)
        RMS_test.append(np.sqrt(2. * E_test / N))
        if rms < test_opt_rms:
            test_opt_rms = rms
            test_opt_m = L
            test_opt_w = w
    print(RMS, "\n", RMS_test)
    plt.plot(range(5, 30, 5), RMS,  c='r', label="training RMS")
    plt.plot(range(5, 30, 5), RMS_test, c='b', label="test RMS")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("Error")
    plt.title("RMS comparison")
    plt.show()

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

    nx = np.linspace(start=0., stop=60., num=1000).reshape(1000, 1)
    phi = PHI(nx, m=opt_m)
    ny = phi.dot(opt_w)

    plt.plot(x, t, label="training data")
    plt.plot(nx, ny, label="plot lin")
    plt.xlabel("time")
    plt.ylabel("acceleration")
    plt.title("best model on training m={}".format(opt_m))
    plt.legend()
    plt.show()

    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])

    nx = np.linspace(start=0., stop=60., num=1000.).reshape(1000, 1)
    phi = PHI(nx, m=test_opt_m)
    ny = phi.dot(test_opt_w)

    plt.plot(x_test, t_test, label="test data")
    plt.plot(nx, ny, label="plot lin")
    plt.xlabel("time")
    plt.ylabel("acceleration")
    plt.title("best model on test m={}".format(test_opt_m))
    plt.legend()
    plt.show()






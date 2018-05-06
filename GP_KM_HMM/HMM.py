#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 4: Part 3 HMM
"""

from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
import scipy.stats
import matplotlib.pyplot as plt


def sim_machine(T):
    loaded = False # fair
    w = ""
    v = ""
    a = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
    for t in range(1000):
        if not loaded:
            w += "F"
            v += str(np.random.randint(1,7))
            if np.random.rand() > .95:
                loaded = True
        else:
            w += "L"
            v += str(a[np.random.randint(0, 10)])
            if np.random.rand() > .9:
                loaded = False
    return w, v


if __name__=="__main__":
    T = 1000
    N = 2
    w, v = sim_machine(T)
    print(w)
    a = np.array([[.95, .10],
                  [.05, .90]])
    b = np.vstack((np.array([1., 1., 1., 1., 1., 1.]) / 6.,
                    np.array([.1, .1, .1, .1, .1, .5])))

    alpha = np.zeros(shape=(N, T))
    alpha[0, 0] = 1.
    alpha[1, 0] = 1.
    for t in range(1, T):
        alpha[0, t] = (alpha[0, t-1] * a[0, 0] + alpha[1, t-1] * a[1, 0]) * b[0, int(v[t])-1]
        alpha[1, t] = (alpha[0, t-1] * a[0, 1] + alpha[1, t-1] * a[1, 1]) * b[1, int(v[t])-1]
        scaling = 1./(alpha[0, t] + alpha[1, t])
        alpha[0, t] *= scaling
        alpha[1, t] *= scaling

    beta = np.zeros(shape=(N, T))
    beta[0, T-1] = 1.
    beta[1, T-1] = 1.
    for t in range(T-2, -1, -1):
        beta[0, t] = (beta[0, t+1] * a[0, 0] + beta[1, t+1] * a[0, 1]) * b[0, int(v[t])-1]
        beta[1, t] = (beta[0, t+1] * a[1, 0] + beta[1, t+1] * a[1, 1]) * b[1, int(v[t])-1]
        scaling = 1. / (beta[0, t] + beta[1, t])
        beta[0, t] *= scaling
        beta[1, t] *= scaling

    final = np.zeros(shape=(N, T))
    for t in range(T):
        final[0, t] = alpha[0, t] * beta[0, t]
        final[1, t] = alpha[1, t] * beta[1, t]
        scaling = 1./np.sum(final[:, t])
        final[:, t] *= scaling

    original = np.zeros(T)
    for i in range(T):
        if w[i] == 'L':
            original[i] = 1.
    plt.plot(range(T), original, 'b-', label='original')
    plt.plot(range(T), alpha[1, :], 'r-', label='forward')
    plt.yticks(np.linspace(0., 1.5, 10.))
    plt.xlabel("Time $t$")
    plt.ylabel("$P(Z_t|Z_{t-1}$)")
    plt.legend()
    plt.title("HMM Forward")
    plt.show()

    plt.plot(range(T), original, 'b-', label='original')
    plt.plot(range(T), beta[1, :], 'g-', label='backward')
    plt.yticks(np.linspace(0., 1.5, 10.))
    plt.xlabel("Time $t$")
    plt.ylabel("$P(Z_t|Z_{t-1}$)")
    plt.title("HMM Backward")
    plt.legend()
    plt.show()

    plt.plot(range(T), original, 'b-', label='original')
    plt.plot(range(T), final[1, :], 'g-', label='combined')
    plt.yticks(np.linspace(0., 1.5, 10.))
    plt.xlabel("Time $t$")
    plt.ylabel("$P(Z_t|Z_{t-1}$)")
    plt.title("HMM Combined")
    plt.legend()
    plt.show()

    plt.savefig("hmm.png", format='png')
#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 4: Part 1 Gaussian Process
"""

from __future__ import print_function
import numpy as np
from scipy.optimize import minimize
import scipy.stats
import matplotlib.pyplot as plt


def PHI_1(xi, xj, sigma):
    """
    :type x: np.ndarray
    :rtype: np.ndarray
    """
    return np.exp(-np.fabs(xi - xj) / sigma)


if __name__ == "__main__":
    training = np.loadtxt("crash.txt")

    N = training.shape[0]
    x = np.array(training[:, 0])
    t = np.array(training[:, 1])

    scaling_factor_x = max(x) - min(x)
    scaling_factor_t = max(t) - min(t)

    print(min(x), scaling_factor_x, scaling_factor_t)
    x = (x - min(x)) / scaling_factor_x
    t = (t - min(t)) / scaling_factor_t
    x_star = np.linspace(x.min(), x.max(), N)
    beta = 1. / (np.std(t)) ** 2.
    """
    For each of these kernel families, construct your Gram matrix K and add diagonal noise to form
    C. In the last assignment, we estimated the beta precision parameter for the noise as 0.0025
    eyeballed the standard deviation sigma = 20, and beta = 1 sigma^2.If you scale sigma by the same magnitude
    as you scaled all of the t values, you can compute the appropriate beta for C. You can now use C, t and the kernel
    function distances between x and each x to predict y
    values at x . First, figure out an appropriate order of magnitude for the sigma parameter. Look at
    the output of your Gaussian process.
    """
    # num = 0
    # for sigma in np.logspace(-.2, .8, num=100):
    #     # from slide# 12 inlec18.pdf joint distribution is
    #     #     | 0   C1    k1 |
    #     # N1 =|   ,          |
    #     #     | 0   k1^T  c1 |
    #     K1 = np.zeros(shape=(N+1, N+1))  # K Gram Matrix or kernel for kernel function PHI_1
    #     print("{} sigma={} = 10^ {:.3f}".format(num, sigma, np.log10(sigma)))
    #     for i in range(N):
    #         for j in range(N):
    #             K1[i, j] = PHI_1(xi=x[i], xj=x[j], sigma=sigma)
    #
    #     t_star = np.zeros(N)
    #     for ti in range(N):
    #         for i in range(N):
    #             K1[i, N] = PHI_1(xi=x[i], xj=x_star[ti], sigma=sigma)
    #             K1[N, i] = PHI_1(xi=x_star[ti], xj=x[i], sigma=sigma)
    #         K1[N, N] = PHI_1(xi=x_star[ti], xj=x_star[ti], sigma=sigma)
    #         C1 = K1[:N, :N] + (1./beta) * np.identity(N)
    #         k1 = K1[:N, N]
    #         c1 = K1[N, N]
    #         m = k1.T.dot(np.linalg.inv(C1)).dot(t)
    #         D = c1 - k1.T.dot(np.linalg.inv(C1)).dot(k1)
    #         t_star[ti] = m #scipy.stats.norm(m, D).pdf(x_star[ti])
    #     plt.plot(x, t, c='r', label="data")
    #     plt.plot(x_star, t_star, c='g', label="k1")
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.title("GP with $lg \sigma$={}".format(np.log10(sigma)))
    #     plt.savefig("fig/{:04d}".format(num), format='png')
    #     plt.clf()
    #     num += 1

    """
    Once you have found a reasonable value of sigma perform five-fold cross-validation on 100 values
    of sigma of the same order of magnitude as your rough calculation found, computing average MSE and
    determining a best-fit hyperparameter value.
    For each of the kernel functions, plot the training data and the output of the Gaussian process
    with the best-fit hyperparameter (by plotting 100 evenly spaced x values and their corresponding
    GP outputs).
    """
    # sigma 0.00037~0.00038 is the somewhat matching the data so be it
    # 5 fold validaiton
    num_of_folds = 5
    fold_size = N / num_of_folds # 18

    num_sigmas = 100
    avg_mse = np.zeros(num_sigmas)
    avg_i = 0
    opt_mse = np.inf
    opt_sigma = np.nan
    for sigma in np.logspace(-.2, .8, num=num_sigmas):
        num = 0
        # print("s={} sigma={} beta={}".format(avg_i, sigma, beta))
        mse = np.zeros(num_of_folds)
        for f in range(num_of_folds):
            mask = np.ones(shape=x.shape[0], dtype="bool")
            mask[f * fold_size:(f + 1) * fold_size] = False
            train_x = x[mask.flatten()]
            train_t = t[mask.flatten()]
            x_star = x[np.logical_not(mask.flatten())]
            t_star = t[np.logical_not(mask.flatten())]

            N_train = train_x.shape[0]
            N_val = x_star.shape[0]
            validation_t_hat = np.zeros(N_val)
            K1 = np.zeros(shape=(N_train + 1, N_train + 1))  # K Gram Matrix or kernel for kernel function PHI_1
            for i in range(N_train):
                for j in range(N_train):
                    K1[i, j] = PHI_1(xi=train_x[i], xj=train_x[j], sigma=sigma)

            for ti in range(N_val):
                for i in range(N_train):
                    K1[i, N_train] = PHI_1(xi=train_x[i], xj=x_star[ti], sigma=sigma)
                    K1[N_train, i] = PHI_1(xi=x_star[ti], xj=train_x[i], sigma=sigma)
                K1[N_train, N_train] = PHI_1(xi=x_star[ti], xj=x_star[ti], sigma=sigma)
                C1 = K1[:N_train, :N_train] + (1. / beta) * np.identity(N_train)
                k1 = K1[:N_train, N_train]
                c1 = K1[N_train, N_train]
                m = k1.T.dot(np.linalg.inv(C1)).dot(train_t)
                D = c1 - k1.T.dot(np.linalg.inv(C1)).dot(k1)
                validation_t_hat[ti] = m  # scipy.stats.norm(m, D).pdf(x_star[ti])
            mse[f] = (.5 * np.linalg.norm(t_star - validation_t_hat) ** 2.) / N_train
        avg_mse[avg_i] = np.average(mse)
        if not np.isnan(np.average(mse)) and np.average(mse) < opt_mse:
            opt_sigma = sigma
        avg_i += 1
    print("optimum avg_mse={} sigma={}".format(np.nanmin(avg_mse), opt_sigma))

    x_star = np.linspace(x.min(), x.max(), N)
    K1 = np.zeros(shape=(N+1, N+1))
    for i in range(N):
        for j in range(N):
            K1[i, j] = PHI_1(xi=x[i], xj=x[j], sigma=sigma)

    t_star = np.zeros(N)
    for ti in range(N):
        for i in range(N):
            K1[i, N] = PHI_1(xi=x[i], xj=x_star[ti], sigma=opt_sigma)
            K1[N, i] = PHI_1(xi=x_star[ti], xj=x[i], sigma=opt_sigma)
        K1[N, N] = PHI_1(xi=x_star[ti], xj=x_star[ti], sigma=opt_sigma)
        C1 = K1[:N, :N] + (1. / beta) * np.identity(N)
        k1 = K1[:N, N]
        c1 = K1[N, N]
        m = k1.T.dot(np.linalg.inv(C1)).dot(t)
        D = c1 - k1.T.dot(np.linalg.inv(C1)).dot(k1)
        t_star[ti] = m

    plt.plot(x, t, c='r', label="data")
    plt.plot(x_star, t_star, c='g', label="k1")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Best fit GP with $\sigma_*$={:.6f}".format(opt_sigma))
    plt.savefig("fig_k2.png", format='png')
    plt.show()



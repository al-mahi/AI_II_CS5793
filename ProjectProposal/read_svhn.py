#!/usr/bin/python

from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    train = loadmat('train_32x32.mat')
    train_x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    train_y = train['y'].flatten()
    print(train_x.shape, train_y.shape)
    plt.suptitle("Plotting SVHN data set using python")
    for i in range(9):
        num = 331+i
        ax = plt.subplot(num)
        ax.imshow(train_x[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(str(train_y[i]))
    plt.show()

    test = loadmat('test_32x32.mat')
    test_x = test['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    test_y = test['y'].flatten()
    print(test_x.shape, test_y.shape)
    plt.suptitle("Plotting SVHN data set using python")
    for i in range(9):
        num = 331 + i
        ax = plt.subplot(num)
        ax.imshow(test_x[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(str(test_y[i]))
    plt.show()

    gnb = GaussianNB()
    y_pred = gnb.fit(train_x, test_y).predict(test_x)



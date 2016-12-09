#!/usr/bin/python

from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB, BaseNB
import numpy as np

from itertools import cycle

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def plot_roc_with_cross_validation(x, y, tx, ty, n_folds=5, description="Plot"):
    # Run classifier with cross-validation and plot ROC curves
    cv = KFold(n_splits=n_folds)
    classifier = GaussianNB()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue'])
    lw = 2

    i = 0
    for (tr, tt), color in zip(cv.split(x, y), colors):
        y_hat = classifier.fit(x[tr], y[tr]).predict(x[tt])
        fpr, tpr, thresholds = roc_curve(y[tt], y_hat, pos_label=9)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1

    accuracy = accuracy_score(ty, classifier.fit(x,y).predict(tx))
    mean_tpr /= cv.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("{} accuracy {:.2%}".format(description, accuracy))
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    train = loadmat('train_32x32.mat')
    x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    y = train['y'].flatten()
    np.place(y, y==10, 0)

    test = loadmat('test_32x32.mat')
    tx = test['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    ty = test['y'].flatten()
    np.place(ty, ty == 10, 0)

    x = np.array(x.reshape(x.shape[0], 32 * 32 * 3))
    tx = np.array(tx.reshape(tx.shape[0], 32 * 32 * 3))

    plot_roc_with_cross_validation(x=x, y=y, tx=tx, ty=ty, n_folds=5, description='ROC for Gaussian Naive Bayes Classifier\nconsidering RGB as independent')

    x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
    tx = test['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)

    x = np.mean(x, axis=3)
    tx = np.mean(tx, axis=3)
    x = np.array(x.reshape(x.shape[0], 32 * 32))
    tx = np.array(tx.reshape(tx.shape[0], 32 * 32))

    plot_roc_with_cross_validation(x=x, y=y, tx=tx, ty=ty, n_folds=5, description='ROC for Gaussian Naive Bayes Classifier\ntaking the average of rgb')

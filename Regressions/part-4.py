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


def flower_to_float(s):
    d = {
        "Iris-setosa": 0.,
        'Iris-versicolor': 1.,
        'Iris-virginica': 2.
    }
    return d[s]


irises = np.loadtxt('iris.data', delimiter=',', converters={4: flower_to_float})

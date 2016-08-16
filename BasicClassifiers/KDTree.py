#!/usr/bin/python

import numpy as np

class KDTree:

    def __init__(self, matrix):
        """
        :param matrix: should be a numpy array where each row is a data element
        :type matrix: np.array
        and each column is a feature. This operation should recursively
        split the data along the median of each feature vector in turn.
        """
        np.array()

    def find_nearest(self, vector):
        """
        :param vector: is a numpy array with a single row representing a single data element.
        This method should traverse the tree to a leaf, then search enough of the tree to guarantee
        finding the neighbor.
        :return: nearest neighbor vector should be returned.
        """
        pass

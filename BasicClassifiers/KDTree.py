#!/usr/bin/python

"""
Author: S M Al Mahi
CS5793: Artificial Intelligence II
Assignment 1: Basic classifiers
#1
kD Tree
Design and code your own kD tree data structure as a Python object. Your tree can be less general
than a standard implementation; you need only account for two data dimensions, and you need
only support 1-nearest-neighbor search.
"""

from __future__ import print_function
import numpy as np
from scipy.spatial import cKDTree


class KDTree:
    def __init__(self, matrix):
        """
        :type matrix: np.ndarray
        :param matrix: should be a numpy array where each row is a data element
        and each column is a feature. This operation should recursively
        split the data along the median of each feature vector in turn.
        """
        self._matrix = matrix
        self._dim = self._matrix.shape[-1]

        def build_kdtree(mat, axis, ancestor):
            """
            :param ancestor: ancestor of current node. been used only for 2d visualization
            :param axis: split axis
            :rtype: np.ndarray
            :type mat: np.ndarray
            """
            if mat.shape[0] == 0:
                return None
            dim = len(mat[0])
            loc = mat.shape[0] / 2
            mat = mat[mat[:, axis].argsort()]
            key = mat[loc]
            # np.delete(mat, loc, 0)
            left = build_kdtree(mat=mat[:loc], axis=(axis + 1) % dim, ancestor=key)
            right = build_kdtree(mat=mat[loc + 1:], axis=(axis + 1) % dim, ancestor=key)  # skipping because deleted 1
            return Node(coordinate=key, left=left, right=right, axis=axis, ancestor=ancestor)

        self._root = build_kdtree(matrix, 0, None)

    def find_nearest(self, vector):
        """
        :type vector: np.ndarray
        :param vector: is a numpy array with a single row representing a single data element.
        This method should traverse the tree to a leaf, then search enough of the tree to guarantee
        finding the neighbor.
        :return: nearest neighbor vector should be returned.
        """
        if vector.shape[0] != self._dim:
            print("Vector dimension is {} expected {}".format(vector.shape[0], self._dim))
            return

        def visit(node, vector, axis, min_dist, nn_coordinate):
            """
            :param axis: split axis 0, 1, ... denotes x, y, ... axis respectively in space
            :param nn_coordinate: coordinate of nearest neighbour
            :param min_dist: minimum distance found so far between vector and a data point
            :param vector: same as find_nearest method
            :type node: Node
            """
            if node is None:
                return min_dist, nn_coordinate
            if node.left is None and node.right is None:
                dist = np.sqrt(sum((vector - node.coordinate) ** 2.))
                if dist < min_dist:
                    nn_coordinate = node.coordinate
                    min_dist = dist
                    return min_dist, nn_coordinate

            dist = np.sqrt(sum((vector - node.coordinate) ** 2.))
            if vector[axis] < node.coordinate[node.axis]:
                min_dist, nn_coordinate = visit(node=node.left, vector=vector, axis=(axis + 1) % vector.shape[0], min_dist=min_dist, nn_coordinate=nn_coordinate)
                if dist < min_dist:
                    nn_coordinate = node.coordinate
                    min_dist = dist
                if np.fabs(vector[axis] - node.coordinate[node.axis]) < min_dist:
                    min_dist, nn_coordinate = visit(node=node.right, vector=vector, axis=(axis + 1) % vector.shape[0],
                                                    min_dist=min_dist, nn_coordinate=nn_coordinate)
            else:
                min_dist, nn_coordinate = visit(node=node.right, vector=vector, axis=(axis + 1) % vector.shape[0],
                                                min_dist=min_dist, nn_coordinate=nn_coordinate)
                if dist < min_dist:
                    nn_coordinate = node.coordinate
                    min_dist = dist
                if np.fabs(vector[axis] - node.coordinate[node.axis]) < min_dist:
                    min_dist, nn_coordinate = visit(node=node.left, vector=vector, axis=(axis + 1) % vector.shape[0],
                                                    min_dist=min_dist, nn_coordinate=nn_coordinate)
            return min_dist, nn_coordinate

        res = visit(node=self._root, vector=vector, axis=0, min_dist=np.inf, nn_coordinate=None)
        return res[1]

    def visualize(self, vector):
        if self._dim > 2:
            print("Visualization available nly for 2d data")
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.suptitle("KDtree")
        ax.scatter(self._matrix[:, 0], self._matrix[:, 1])
        ax.scatter(vector[0], vector[1], c='red')
        for i in range(self._matrix.shape[0]):
            ax.annotate("{:.0f},{:.0f}".format(self._matrix[i, 0], self._matrix[i, 1]), (self._matrix[i, 0], self._matrix[i, 1]))

        # visit
        def visit(node):
            """
            :type node: Node
            """
            xmin = ymin = 0.
            xmax = max(self._matrix[:, 0]) + 10
            ymax = max(self._matrix[:, 1]) + 10
            if node.axis == 0:
                if node.ancestor is not None:
                    if node.ancestor[1] < node.coordinate[1]:
                        ymin = node.ancestor[1]
                    else:
                        ymax = node.ancestor[1]
                plt.plot((node.coordinate[0], node.coordinate[0]), (ymin, ymax), 'r-')
            if node.axis == 1:
                if node.ancestor[0] < node.coordinate[0]:
                    xmin = node.ancestor[0]
                else:
                    xmax = node.ancestor[0]
                plt.plot((xmin, xmax), (node.coordinate[1], node.coordinate[1]), 'b-')
            if node.left is not None:
                visit(node.left)
            if node.right is not None:
                visit(node.right)

        visit(self._root)
        plt.show()


class Node:
    def __init__(self, coordinate, left, right, axis, ancestor):
        """
        :type coordinate: np.ndarray
        :param coordinate: coordination of this node
        :type left: np.ndarray
        :param left: coordination of left child
        :type right: np.ndarray
        :param right: coordination of right
        """
        self._coordinate = coordinate
        self._left = left
        self._right = right
        self._axis = axis
        self._ancestor = ancestor  # for visualizing

    @property
    def coordinate(self):
        return self._coordinate

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def ancestor(self):
        return self._ancestor

    @property
    def axis(self):
        return self._axis

    def __str__(self):
        return "[" + str(self._coordinate) + " " + self._axis + "(" + str(self._left) + ") {" + str(self._right) + "}]"

if __name__ == "__main__":
    def test(data, vector=None):
        if vector is None:
            vector = np.random.uniform(low=0., high=100.0, size=data.shape[1])
        print("input data {}:\n{}".format(data.shape, data))
        print("vector: {}".format(vector))
        kdt = KDTree(data)
        print("My KD NN: {}".format(kdt.find_nearest(vector)))
        from scipy.spatial import cKDTree
        s_tree = cKDTree(data)
        scipy_res = s_tree.query(vector, k=1)
        print("Scipy NN: {}".format(s_tree.data[scipy_res[1]]))
        # kdt.visualize(vector=vector)

    data = np.random.uniform(low=0., high=10000., size=(10000, 5))
    test(data)

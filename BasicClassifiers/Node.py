#!/usr/bin/python


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
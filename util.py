import numpy as np


def get_first_one_1d(array):
    """
    Returns the index of the first one in the row of the 1d array.
    """
    return np.where(array == 1)[0][0]


def get_first_one_2d(array, row):
    """
    Returns the index of the first one in the row of the 2d array.
    """
    return np.where(array[row] == 1)[0][0]

import os
import structlog
import logging
import numpy as np


structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if bool(os.getenv('DEBUG')) else logging.INFO)
)

logger = structlog.get_logger()


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


def get_first_card(turn_cards, turn_index, starting_index):
    """
    Takes in a list of cards and returns the first card played.
    If the current turn (turn_index) is the first turn (starting_index),
    then return None because no cards have been played yet.
    """
    return turn_cards[starting_index] if turn_index != 0 else None

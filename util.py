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


def get_suit(card: int):
    return card // 13


def is_card_better(test_card: int, base_card: int):
    """
    Error if the two cards are not the same suit.
    Returns True if the test_card is better than the base_card.
    """
    assert get_suit(test_card) == get_suit(base_card)
    return (test_card - 1) % 13 > (base_card - 1) % 13  # account for Ace = 0 > King = 12
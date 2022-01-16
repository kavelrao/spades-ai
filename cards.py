import numpy as np
from enum import IntEnum

from util import get_first_one_2d


"""
Implements objects that may be common to multiple card games
"""


class Suits(IntEnum):
    BLANK = -1
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


CARD_MAP = {
    0: 'Two',
    1: 'Three',
    2: 'Four',
    3: 'Five',
    4: 'Six',
    5: 'Seven',
    6: 'Eight',
    7: 'Nine',
    8: 'Ten',
    9: 'Jack',
    10: 'Queen',
    11: 'King',
    12: 'Ace'
}


class Bid:
    def __init__(self, value):
        if value < -1 or value > 13:
            raise AttributeError("Bid() argument 'value' out of valid range [0, 13]")
        self.array = np.zeros((1, 14))
        self.value = value
        if value > -1:
            self.array[0, value] = 1

    @classmethod
    def from_array(cls, array):
        """
        Initializes a Bid from a representative array
        """
        try:
            if array.shape == (1, 14):
                value = get_first_one_2d(array, 0)
                return cls(value)
            else:
                raise TypeError("from_array() argument 'array' is invalid shape; must be (1, 14)")
        except AttributeError:
            raise TypeError("from_array() argument 'array' is missing or invalid type")
        
    @staticmethod
    def list_to_np(bids, n=4):
        """
        Converts a list of n bids to a (n, 14) array where each row is a one-hot vector of the bid
        """
        arr = np.zeros((n, 14))
        for i, bid in enumerate(bids):
            arr[i] = bid.array
        return arr

    def __repr__(self):
        if self.value == -1:
            return 'X'
        return str(self.value)

    # Comparison operators
    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


class Card:
    def __init__(self, value):
        if value < -1 or value > 51:
            raise AttributeError("Card() argument 'value' out of valid range [-1 -51]")
        self.array = np.zeros((1, 52))
        self.value = value
        if value > -1:
            self.array[0, value] = 1

    @classmethod
    def from_array(cls, array):
        """
        Initializes a Card from a representative array
        """
        try:
            if array.shape == (1, 52):
                value = get_first_one_2d(array, 0)
                return cls(value)
            else:
                raise TypeError("from_array() argument 'array' is invalid shape; must be (1, 52)")
        except AttributeError:
            raise TypeError("from_array() argument 'array' is missing or invalid type")

    def suit(self) -> Suits:
        return Suits(self.value // 13)

    def is_better(self, other):
        if self.suit() == other.suit():
            return self.value % 13 > other.value % 13
        else:
            return self.suit() == Suits['SPADES']

    def is_valid_play(self, other, hand):
        if self.suit() == other.suit():
            return True
        return not hand.has_suit(other.suit())

    @staticmethod
    def list_to_np(cards, n=13):
        """
        Converts a list of n cards to a (n, 52) array where each row is a one-hot vector of the card
        """
        arr = np.zeros((n, 52))
        for i, card in enumerate(cards):
            arr[i] = card.array
        return arr

    def __repr__(self):
        if self.value == -1:
            return 'Blank Card'
        suit = str(self.suit())  # 'Suits.SUIT'
        suit = suit.split('.')[1]  # 'SUIT'
        suit = suit[0] + suit[1:].lower()  # 'Suit'
        name = CARD_MAP[self.value % 13]
        return f'{name} of {suit}'

    # Comparison operators
    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


class Hand:
    def __init__(self):
        self.array = np.zeros((1, 52))
        self.cards = list()

    def deal(self, card):
        if card not in self.cards:
            self.cards.append(card)
            self.array[0, card.value] = 1
        else:
            raise AttributeError("deal() argument 'card' was already in this hand")

    def sort(self):
        self.cards = sorted(self.cards)

    def has_card(self, card):
        return any(my_card == card for my_card in self.cards)

    def has_suit(self, suit):
        return any(card.suit() == suit for card in self.cards)

    def play_card(self, card):
        if not self.has_card(card):
            raise AttributeError("play_card() argument 'card' is not in this hand")
        self.cards.remove(card)
        self.array[0, card.value] = 0
        return card

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, item):
        return self.cards[item]
    
    def __repr__(self):
        return str(self.cards)

import numpy as np

from util import get_first_one_2d


class AgentBase:
    """
    The skeleton for a Spades Agent
    """
    def deal(self, hand):
        """
        Sets hand to be the dealt multi-hot (1, 52) array
        """
        self.hand = hand.reshape((1, 52))

    def get_bid(self, bid_state):
        """
        Returns the bid as a (1, 14) one-hot vector where the index represents the bid amount
        """
        pass

    def get_play(self, bids, scores, previous_play, turn_cards):
        """
        Returns the card to play as a (1, 52) one-hot vector where the index represents the card
        """
        pass


class DummyAgent(AgentBase):
    """
    The simplest possible Spades Agent
    """
    def get_bid(self, state):
        bid = np.zeros((1, 14))
        bid[0, 3] = 1  # always bid 3
        return bid
    
    def get_play(self, bids, scores, previous_play, turn_cards):
        play = np.zeros((1, 52))
        card_index = get_first_one_2d(self.hand, 0)
        play[0, card_index] = 1
        self.hand[0, card_index] = 0
        return play

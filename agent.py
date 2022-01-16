import numpy as np

from spades import Bid, Card, Hand
from util import get_first_one_2d


class AgentBase:
    """
    The skeleton for a Spades Agent
    """
    def __init__(self):
        self.hand = None
        self.player_id = None

    def deal(self, hand, player_id):
        """
        Sets hand to be the dealt Hand
        """
        self.hand = hand
        self.hand.sort()
        self.player_id = player_id

    def get_bid(self, bid_state):
        """
        Returns the bid as a Bid in which the (1, 14) one-hot vector where the index represents the bid amount
        """
        pass

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, suit):
        """
        Returns the card to play as a (1, 52) one-hot vector where the index represents the card
        and removes the card from the player's hand
        """
        pass


class DummyAgent(AgentBase):
    """
    The simplest possible Spades Agent
    Always bids 3 and plays the lowest valid card
    """
    def get_bid(self, bid_state):
        return Bid(3)
    
    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, suit):
        if turn_index == 0 or not self.hand.has_suit(suit):
            return self.hand.play_card(self.hand[0])

        # if it needs to match the suit
        for card in self.hand.cards:
            if card.suit() == suit:
                return self.hand.play_card(card)


class UserAgent(AgentBase):
    """
    A user-controlled Spades Agent
    """
    def get_bid(self, bid_state):
        print('Current bids:')
        print(bid_state)
        print('Your hand:')
        print(self.hand)
        bid_num = input('Enter bid: ')
        try:
            bid_num = int(bid_num)
        except ValueError:
            bid_num = -1
        while bid_num < 0 or bid_num > 13:
            print('Bid must be in range 0 - 13')
            bid_num = input('Enter bid: ')
            try:
                bid_num = int(bid_num)
            except ValueError:
                bid_num = -1
        print()
        return Bid(bid_num)

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, suit):
        print(f'You are player {self.player_id} with turn {turn_index} in the round')
        print('Current scores:')
        print(scores[-1])
        print('Bids:')
        print(bids)
        print('Previous play:')
        print(previous_play)
        print('Current plays:')
        print(turn_cards)
        print('Your hand:')
        print(self.hand)
        play_index = int(input('Enter index of card to play: '))
        while play_index < 0 or play_index > len(self.hand):
            print('Play must be in your hand')
            play_index = int(input('Enter index of card to play: '))
        print()
        return self.hand.play_card(self.hand[play_index])

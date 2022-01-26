import numpy as np

from cards import Bid, Card, Hand
from util import get_first_card


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

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, starting_index, spades_broken):
        """
        Returns the card to play as a (1, 52) one-hot vector where the index represents the card
        and removes the card from the player's hand
        """
        pass


class TrainedAgent(AgentBase):
    """
    The skeleton for an algorithmically trained Spades Agent
    Must define a training method
    """
    
    @classmethod
    def train(cls, *args, **kwargs):
        pass


class DummyAgent(AgentBase):
    """
    The simplest possible Spades Agent
    Always bids 3 and plays the lowest valid card
    """

    def get_bid(self, bid_state):
        return Bid(3)
    
    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, starting_index, spades_broken):
        # if we don't need to match suit, play the first card
        if turn_index == 0 or not self.hand.has_suit(turn_cards[starting_index].suit()):
            return self.hand.play_card(self.hand[0])

        # if we need to match suit, play first card that is valid suit
        for card in self.hand.cards:
            if card.suit() == turn_cards[starting_index].suit():
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

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, starting_index, spades_broken):
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
        first_card = get_first_card(turn_cards, turn_index, starting_index)
        while play_index < 0 or play_index > len(self.hand) or not self.hand[play_index].is_valid_play(self.hand, spades_broken, first_card, check_hand=True):
            print('Play must be in your hand and valid according to the rules')
            play_index = int(input('Enter index of card to play: '))
        print()
        return self.hand.play_card(self.hand[play_index])

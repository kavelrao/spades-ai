import numpy as np
from enum import IntEnum

from util import get_first_one_2d, get_suit, is_card_better


class Suits(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Spades:
    def __init__(self, players):
        self.players = players
        self.dealer_player = 0
        self.starting_player = 1
        self.bids = np.ndarray((0, 4, 14))  # bids of shape (4, 14) grouped into rounds
        self.tricks = np.ndarray((0, 13, 1, 4))  # tricks of shape (1, 4) grouped into rounds
        self.cards_played = np.ndarray((0, 13, 4, 52))  # cards played of shape (4, 52) grouped into rounds
        self.scores = np.ndarray((0, 1, 2))  # running scores of shape (1, 2) grouped into rounds

    def deal(self):
        player_hands = np.zeros((4, 52))
        deck_cards = list(range(52))
        for _ in range(13):
            for player in range(4):
                card_index = np.random.randint(0, len(deck_cards))
                card = deck_cards[card_index]
                player_hands[player, card] = 1
                deck_cards.pop(card_index)
        for i, player in enumerate(self.players):
            player.deal(player_hands[i])

    def bid(self):
        bid_state = np.zeros((4, 14))  # for each player, one-hot vector for the bid value
        player_order = self.players[self.starting_player:] + self.players[:self.starting_player]
        for i, player in enumerate(player_order):
            bid_state[i] = player.get_bid(bid_state)
        return bid_state

    def turn(self, bids, previous_play):
        print(f'starting player: {self.starting_player}')
        turn_cards = np.zeros((4, 52))  # for each player, one-hot vector for the card played
        played_cards = np.zeros(4)  # for each player, a number representing the card played. used to calculate winning card
        winning_card = 0
        winning_player = self.starting_player  # first card played is automatically "winning" before any other plays
        trick = np.zeros((1, 4))  # set to one for the player that wins the trick
        player_order = self.players[self.starting_player:] + self.players[:self.starting_player]
        for i, player in enumerate(player_order):
            player_id = (i + self.starting_player) % 4  # store plays in order of the player ID's
            turn_cards[player_id] = player.get_play(bids, self.scores, previous_play, turn_cards)
            played_cards[player_id] = get_first_one_2d(turn_cards, player_id)  # gets index of first 1 in the returned one-hot array

            if player_id == self.starting_player:
                winning_card = played_cards[player_id]
            else:  # find if the recent card beats the winning card
                if get_suit(played_cards[player_id]) == get_suit(winning_card):
                    if is_card_better(played_cards[player_id], winning_card):
                        winning_card = played_cards[player_id]
                        winning_player = player_id
                elif get_suit(played_cards[player_id]) == Suits['SPADES']:  # if the recent card is a spade trumping a non-spade
                    winning_card = played_cards[player_id]
                    winning_player = player_id

        trick[0, winning_player] = 1
        print('played cards:')
        print(played_cards)
        return trick, turn_cards

    def round(self):
        self.deal()
        self.starting_player = (self.dealer_player + 1) % 4
        round_bids = self.bid()
        round_tricks = np.zeros((0, 1, 4))
        round_cards = np.zeros((0, 4, 52))
        round_score = self.scores[-1].copy().reshape((1, 2)) if len(self.scores) > 0 else np.zeros((1, 2))  # shape of (1, 2)
        # initialize turn state to 0's for first turn
        turn_tricks = np.zeros((1, 4))
        turn_cards = np.zeros((4, 52))
        for turn in range(13):
            turn_tricks, turn_cards = self.turn(round_bids, turn_cards)  # feed in bid and previous turn info
            round_tricks = np.concatenate((round_tricks, turn_tricks.reshape((1, 1, 4))))
            round_cards = np.concatenate((round_cards, turn_cards.reshape((1, 4, 52))))
        
        for team in range(2):
            team_score = 0
            team_bags = 0
            prev_score = self.scores[-1, 0, team] if len(self.scores) > 0 else 0
            prev_bags = prev_score % 10
            p1_bid = get_first_one_2d(round_bids, team)
            p2_bid = get_first_one_2d(round_bids, team + 2)
            team_bid = p1_bid + p2_bid
            p1_tricks = np.sum(round_tricks[:, 0, team])
            p2_tricks = np.sum(round_tricks[:, 0, team + 2])
            team_tricks = p1_tricks + p2_tricks
            # account for null bids
            if p1_bid == 0:
                if p1_tricks > 0:
                    team_score -= 100
                    team_bags += p1_tricks
                team_tricks -= p1_tricks  # any tricks on a null bid go straight to bags
            if p2_bid == 0:
                if p2_tricks > 0:
                    team_score -= 100
                    team_bags += p2_tricks
                team_tricks -= p2_tricks

            if team_tricks >= team_bid:
                team_score += team_bid * 10
                team_bags += team_tricks - team_bid
            else:
                team_score -= team_bid

            team_score += team_bags
            if prev_bags + team_bags >= 10:
                team_score -= 100

            round_score[0, team] += team_score

        # update game state
        self.bids = np.concatenate((self.bids, round_bids.reshape((1, 4, 14))))
        self.scores = np.concatenate((self.scores, round_score.reshape((1, 1, 2))))
        self.tricks = np.concatenate((self.tricks, round_tricks.reshape((1, 13, 1, 4))))
        self.cards_played = np.concatenate((self.cards_played, round_cards.reshape(1, 13, 4, 52)))
        self.dealer_player = (self.dealer_player + 1) % 4

    def game(self):
        round = 0
        while len(self.scores) == 0 or (np.max(self.scores[-1]) - np.min(self.scores[-1]) < 500 and np.max(self.scores[-1]) < 500):
            self.round()
            print(self.scores[-1])
            round += 1
            if round > 100:
                print('Exceeded max rounds')
                break


"""
cards: 0 - 12: Ace - King of Clubs; 13 - 25: Ace - King of Diamonds; 26 - 38: Ace - King of Hearts; 39 - 51: Ace - King of Spades
"""

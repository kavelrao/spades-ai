import numpy as np

from cards import Bid, Card, Hand, Suits
from util import get_first_one_2d
from agent import AgentBase


BLANK_BID = Bid(-1)
BLANK_CARD = Card(-1)


class Spades:
    NUM_PLAYERS = 4

    def __init__(self, players, null_points: int = 100, win_points: int = 500, max_rounds: int = 1000):
        self.players = players
        if len(self.players) != Spades.NUM_PLAYERS:
            raise AttributeError("Players parameter must have length 4")
        if any(not isinstance(player, AgentBase) for player in players):
            raise AttributeError("Players must extend AgentBase")

        # Game properties
        self.null_points = null_points
        self.win_points = win_points
        self.max_rounds = max_rounds

        self.dealer_player = 0
        self.starting_player = 1
        self.bids = list()  # list of lists of bids, split into rounds
        self.tricks = np.ndarray((0, Hand.HAND_LEN, 1, Spades.NUM_PLAYERS))  # tricks of shape (1, 4) grouped into rounds
        self.cards_played = list()  # list of lists of lists of cards, split into rounds then turns
        self.scores = np.zeros((1, 1, 2))  # running scores of shape (1, 2) grouped into rounds
        self.spades_broken = False

    def deal(self):
        player_hands = [Hand(), Hand(), Hand(), Hand()]
        deck_cards = list(range(Card.CARD_LEN))
        for _ in range(Hand.HAND_LEN):
            for player in range(Spades.NUM_PLAYERS):
                card_index = np.random.randint(0, len(deck_cards))
                card = deck_cards[card_index]
                player_hands[player].deal(Card(card))
                deck_cards.pop(card_index)
        for i, player in enumerate(self.players):
            player.deal(player_hands[i], i)

    def bid(self):
        bid_state = [BLANK_BID] * Spades.NUM_PLAYERS
        player_order = self.players[self.starting_player:] + self.players[:self.starting_player]
        for i, player in enumerate(player_order):
            bid_state[i] = player.get_bid(bid_state)
        return bid_state

    def turn(self, bids, previous_play):
        print(f'Starting player: {self.starting_player}')
        played_cards = [BLANK_CARD] * Spades.NUM_PLAYERS
        first_card = BLANK_CARD
        winning_card = BLANK_CARD
        winning_player = self.starting_player  # first card played is automatically "winning" before any other plays
        trick = np.zeros((1, Spades.NUM_PLAYERS))  # set to one for the player that wins the trick
        player_order = self.players[self.starting_player:] + self.players[:self.starting_player]
        for i, player in enumerate(player_order):
            player_id = player.player_id  # store plays in order of the player ID's
            new_card = player.get_play(i, bids, self.scores, previous_play, played_cards, first_card.suit())

            if player.hand.has_card(new_card):
                raise AttributeError(f"Card played by player id {player_id} is still in their hand")
            if player_id != self.starting_player and not new_card.is_valid_play(first_card, player.hand):
                raise ValueError(f"Card played by player id {player_id} is invalid")
            if new_card.suit() == Suits['SPADES'] and not self.spades_broken:
                if player_id == self.starting_player:
                    if any(player.hand.has_suit(suit) for suit in [Suits['CLUBS'], Suits['DIAMONDS'], Suits['HEARTS']]):
                        raise ValueError(f"Card played by player id {player_id} illegally broke spades")
                self.spades_broken = True

            played_cards[player_id] = new_card

            if player_id == self.starting_player:
                first_card = new_card
                winning_card = new_card
            elif new_card.is_better(winning_card):
                    winning_card = new_card
                    winning_player = player_id

        trick[0, winning_player] = 1
        print('Played cards:')
        print(played_cards)
        print(f'Player {winning_player} takes the trick with the {winning_card}')
        print()
        return trick, played_cards

    def round(self):
        self.deal()
        self.starting_player = (self.dealer_player + 1) % Spades.NUM_PLAYERS
        self.spades_broken = False
        round_bids = self.bid()
        round_tricks = np.zeros((0, 1, Spades.NUM_PLAYERS))
        round_cards = list()
        round_score = self.scores[-1].copy().reshape((1, 2))
        # initialize turn state for first turn
        turn_tricks = np.zeros((1, Spades.NUM_PLAYERS))
        turn_cards = [BLANK_CARD] * Spades.NUM_PLAYERS

        for turn in range(Hand.HAND_LEN):
            turn_tricks, turn_cards = self.turn(round_bids, turn_cards)  # feed in bid and previous turn info
            round_tricks = np.concatenate((round_tricks, turn_tricks.reshape((1, 1, Spades.NUM_PLAYERS))))
            round_cards.append(turn_cards)

            winner = get_first_one_2d(turn_tricks, 0)
            self.starting_player = winner
        
        for team in range(2):
            team_score = 0
            team_bags = 0
            prev_score = self.scores[-1, 0, team]
            prev_bags = prev_score % 10
            p1_bid = round_bids[team].value  # get_first_one_2d(round_bids, team)
            p2_bid = round_bids[team + 2].value  # get_first_one_2d(round_bids, team + 2)
            team_bid = p1_bid + p2_bid
            p1_tricks = np.sum(round_tricks[:, 0, team])
            p2_tricks = np.sum(round_tricks[:, 0, team + 2])
            team_tricks = p1_tricks + p2_tricks
            # account for null bids
            if p1_bid == 0:
                if p1_tricks == 0:
                    team_score += self.null_points
                else:
                    team_score -= self.null_points
                    team_bags += p1_tricks
                    team_tricks -= p1_tricks  # any tricks on a null bid go straight to bags
            if p2_bid == 0:
                if p2_tricks == 0:
                    team_score += self.null_points
                else:
                    team_score -= self.null_points
                    team_bags += p2_tricks
                    team_tricks -= p2_tricks  # any tricks on a null bid go straight to bags

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
        self.bids.append(round_bids)
        self.scores = np.concatenate((self.scores, round_score.reshape((1, 1, 2))))
        self.tricks = np.concatenate((self.tricks, round_tricks.reshape((1, Hand.HAND_LEN, 1, Spades.NUM_PLAYERS))))
        self.cards_played.append(round_cards)
        self.dealer_player = (self.dealer_player + 1) % Spades.NUM_PLAYERS

    def game(self):
        round = 0
        # game continues until score difference is > 500 or max score is > 500
        while np.max(self.scores[-1]) - np.min(self.scores[-1]) < self.win_points and np.max(self.scores[-1]) < self.win_points:
            self.round()
            print(self.scores[-1])
            round += 1
            if round > self.max_rounds:
                print('Exceeded max rounds')
                break
        print(f'Game finished after {round} rounds')

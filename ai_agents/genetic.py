import numpy as np
import random
import multiprocessing

from agent import AgentBase, TrainedAgent
from cards import Bid, Card, Hand
from util import get_first_card, get_first_one_2d
from spades import Spades


class ConstantWeightsGenetic(TrainedAgent):
    """
    Each agent has a set of weights for bidding and for playing.
    To select an action, it randomly selects according to its weights.

    The agent trains through a genetic evolution method, where each iteration,
    N agents play each other for G games. Then, the X best agents are retained,
    and the remaining (N - X) are replenished through crossover of the weights of the X best.
    """

    def __init__(self,  bid_weights=None, play_weights=None, bid_weights_file: str = None, play_weights_file: str = None):
        """
        Prioritizes passed in weight arrays over filename strings
        """
        super().__init__()

        self.rng = np.random.default_rng()
        self.win_count = 0  # used for genetic evolution training algorithm

        if bid_weights is not None:
            self.bid_weights = bid_weights
        elif bid_weights_file is not None:
            self.bid_weights = np.load(bid_weights_file)
            if self.bid_weights.shape != (1, Bid.BID_LEN):
                raise AttributeError(f"bid weights file must encode a (1, {Bid.BID_LEN}) array")
        else:
            self.bid_weights = self.rng.random((1, Bid.BID_LEN))
            # self.bid_weights = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # help it out by starting at 3

        if play_weights is not None:
            self.play_weights = play_weights
        elif play_weights_file is not None:
            self.play_weights = np.load(play_weights_file)
            if self.play_weights.shape != (1, Card.CARD_LEN):
                raise AttributeError(f"play weights file must encode a (1, {Card.CARD_LEN}) array")
        else:
            self.play_weights = self.rng.random((1, Card.CARD_LEN))

    def get_bid(self, bid_state):
        """
        Samples weighted random values and chooses the largest
        """
        choice_weights = self.rng.random((1, Bid.BID_LEN)) * self.bid_weights
        bid_num = np.argmax(choice_weights)

        return Bid(bid_num)

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, starting_index, spades_broken):
        """
        Returns the card to play as a (1, 52) one-hot vector where the index represents the card
        and removes the card from the player's hand
        """
        first_card = get_first_card(turn_cards, turn_index, starting_index)

        choice_weights = self.rng.random((1, Card.CARD_LEN)) * self.play_weights
        max_index = np.argmax(choice_weights)
        play_card = Card(max_index)
        # select the highest probability card that is a valid play
        while not play_card.is_valid_play(self.hand, spades_broken, first_card, check_hand=True):
            choice_weights[0, max_index] = 0
            max_index = np.argmax(choice_weights)
            play_card = Card(max_index)

        return self.hand.play_card(play_card)


    @classmethod
    def train(cls, population_size: int = 1000, select_number: int = 20, games_per_gen: int = 100, num_generations: int = 1000, num_validation_games: int = 100, mutate_threshold: int = 0.05):
        """
        One generation per game; only the winners continue to the next generation
        """
        rng = np.random.default_rng()

        if population_size % 4 != 0:
            raise AttributeError("population size must be a multiple of 4")
        
        # initialize the first population of agents
        agents = list()
        agents_history = list()
        for _ in range(population_size):
            agents.append(cls())
        agents_history.append(agents)

        for gen_num in range(num_generations):
            print(f'Generation {gen_num}')
            for round_num in range(games_per_gen):
                print(f'Round {round_num}')
                random.shuffle(agents)
                queue = multiprocessing.Queue()
                jobs = []

                for game_num in range(population_size // 4):
                    agent_offset = game_num * 4
                    players = agents[agent_offset:agent_offset + 4]
                    process = multiprocessing.Process(target=ConstantWeightsGenetic.multiprocess_training, args=(queue, agent_offset, players, 100))
                    process.start()
                    jobs.append(process)
                for process in jobs:
                    process.join()

                while not queue.empty():
                    results = queue.get()
                    agent_offset = results.get('pid')
                    if results.get('winning_players') is not None:
                        for index in results.get('winning_players'):
                            agents[agent_offset + index].win_count += 1


            agents_history.append(agents)

            # crossover to repopulate
            # mix weights by selecting one if random is below threshold, or choose the other for each weight
            winning_agents = sorted(agents, key=lambda x: x.win_count, reverse=True)[:select_number]  # choose the best ones to keep and repopulate
            print(f'Best win rate: {winning_agents[0].win_count} / {games_per_gen}')
            agents = winning_agents.copy()
            while len(agents) < population_size:
                first_index = rng.integers(len(winning_agents))
                while second_index := rng.integers(len(winning_agents)) == first_index:
                    pass
                first = agents[first_index]
                second = agents[second_index]
                
                cross_threshold = rng.random()

                new_bid_weights = np.zeros((1, Bid.BID_LEN))
                for i in range(Bid.BID_LEN):
                    if rng.random() < mutate_threshold:
                        new_bid_weights[0, i] = rng.random()
                    else:
                        new_bid_weights[0, i] = first.bid_weights[0, i] if rng.random() < cross_threshold else second.bid_weights[0, i]

                new_play_weights = np.zeros((1, Card.CARD_LEN))
                for i in range(Card.CARD_LEN):
                    if rng.random() < mutate_threshold:
                        new_play_weights[0, i] = rng.random()
                    else:
                        new_play_weights[0, i] = first.play_weights[0, i] if rng.random() < cross_threshold else second.play_weights[0, i]
                
                agents.append(cls(bid_weights = new_bid_weights, play_weights=new_play_weights))

            for agent in winning_agents:
                agent.win_count = 0
            winning_agents.clear()

        # after final evolution, run a number of games and output the weights with the highest win rate
        agents = winning_agents
        for gen_num in range(num_validation_games):
            print(f'Validation game {gen_num}')
            random.shuffle(agents)
            for game_num in range(population_size // 4):
                agent_offset = game_num * 4
                print(f'{agent_offset=}, {len(winning_agents)=}')
                spades_game = Spades(agents[agent_offset:agent_offset + 4])
                results = spades_game.game()
                for index in results.get('winning_players'):
                    agents[agent_offset + index].win_count += 1

        most_wins = agents[0].win_count
        best_agent = agents[0]
        for agent in agents:
            if agent.win_count > most_wins:
                most_wins = agent.win_count
                best_agent = agent

        print(f'Best agent had a win rate of {best_agent.win_count}/{num_validation_games}')

        with open('output/stats.txt', 'w+') as f:
            f.write(f'WIN_RATE: {best_agent.win_count} / {num_validation_games}')
        with open('output/constant_weights_genetic__bid_weights', 'wb') as f:
            np.save(f, best_agent.bid_weights)
        with open('output/constant_weights_genetic__play_weights', 'wb') as f:
            np.save(f, best_agent.play_weights)
    
    @staticmethod
    def multiprocess_training(queue, pid, players, max_rounds):
        spades_game = Spades(players, max_rounds=max_rounds)
        result = spades_game.game()
        result['pid'] = pid
        queue.put(result)

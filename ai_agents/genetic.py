import os
import ujson
import numpy as np
import queue
import multiprocessing

from agent import AgentBase, TrainedAgent
from cards import Bid, Card, Hand
from util import get_first_card, get_first_one_2d, logger
from spades import Spades, multiprocess_spades_game


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
            # self.bid_weights = self.rng.random((1, Bid.BID_LEN))
            self.bid_weights = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # force 3 bid every time

        if play_weights is not None:
            self.play_weights = play_weights
        elif play_weights_file is not None:
            self.play_weights = np.load(play_weights_file)
            if self.play_weights.shape != (1, Card.CARD_LEN):
                raise AttributeError(f"play weights file must encode a (1, {Card.CARD_LEN}) array")
        else:
            self.play_weights = self.rng.random((1, Card.CARD_LEN))

        # make sure no weights are zero, or infinite loops could happen when using argmax
        for i in np.where(self.bid_weights[0] == 0)[0]:
            self.bid_weights[0, i] = np.nextafter(0, 1)
        for i in np.where(self.play_weights[0] == 0)[0]:
            self.play_weights[0, i] = np.nextafter(0, 1)
        
        # disable NIL bid
        self.bid_weights[0, 0] = 0
        

    def get_bid(self, bid_state):
        """
        Chooses the largest weight preference
        """
        bid_num = np.argmax(self.bid_weights)

        return Bid(bid_num)

    def get_play(self, turn_index, bids, scores, previous_play, turn_cards, starting_index, spades_broken):
        """
        Returns the card to play as a (1, 52) one-hot vector where the index represents the card
        and removes the card from the player's hand
        """
        first_card = get_first_card(turn_cards, turn_index, starting_index)

        choice_weights = self.play_weights.copy()
        max_index = np.argmax(choice_weights)
        play_card = Spades.CARD_BANK[max_index]
        # select the highest probability card that is a valid play
        while not play_card.is_valid_play(self.hand, spades_broken, first_card, check_hand=True):
            choice_weights[0, max_index] = 0
            max_index = np.argmax(choice_weights)
            play_card = Spades.CARD_BANK[max_index]

        return self.hand.play_card(play_card)


    @classmethod
    def train(cls, population_size: int = 64, select_number: int = 8, games_per_gen: int = 100, num_generations: int = 1000, num_validation_games: int = 100,
              mutate_threshold: float = 0.1, perturb_mult: float = 0.1, max_rounds: int = 25, output_folder: str = 'output', core_count: int = 4):
        """
        One generation per game; only the winners continue to the next generation
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        config = dict(
            agent_class=str(cls),
            population_size=population_size,
            select_number=select_number,
            games_per_gen=games_per_gen,
            num_generations=num_generations,
            num_validation_games=num_validation_games,
            mutate_threshold=mutate_threshold,
            perturb_mult=perturb_mult,
            max_rounds=max_rounds,
            output_folder=output_folder,
            core_count=core_count,
        )
        with open(f'{output_folder}/config.json', 'w') as f:
            ujson.dump(config, f, indent=4)

        rng = np.random.default_rng()

        if population_size % 4 != 0:
            raise AttributeError("population size must be a multiple of 4")
        if population_size < select_number * 2:
            raise AttributeError("select number must be < 1/2 of population size")

        # initialize the first population of agents
        agents = list()
        for _ in range(population_size):
            agents.append(cls())

        for gen_num in range(num_generations):
            logger.info(f'Starting generation', generation=gen_num)
            for round_num in range(games_per_gen):
                logger.info('Starting self-play round', round_num=round_num)
                rng.shuffle(agents)
                mp_queue = multiprocessing.Queue()
                compiled_results = queue.Queue()
                jobs = []
                for game_num in range(population_size // 4):
                    agent_offset = game_num * 4
                    players = agents[agent_offset:agent_offset + 4]
                    #! Uncomment below and comment process stuff to disable multiprocessing
                    # spades_game = Spades(players, max_rounds=20)
                    # result = spades_game.game()
                    # result['pid'] = agent_offset
                    # compiled_results.put(result)
                    process = multiprocessing.Process(target=multiprocess_spades_game, args=(mp_queue, agent_offset, players), kwargs=dict(max_rounds=max_rounds))
                    jobs.append(process)
                    process.start()

                    # wait for core_count processes at a time since only that many can run simultaneously
                    if len(jobs) == core_count or len(jobs) == population_size // 4:
                        [compiled_results.put(mp_queue.get()) for process in jobs]
                        for process in jobs:
                            process.join()
                            logger.debug('process terminated', exitcode=process.exitcode)
                        jobs.clear()

            while not compiled_results.empty():
                results = compiled_results.get()
                agent_offset = results.get('pid')
                if results.get('winning_players') is not None:
                    for index in results.get('winning_players'):
                        agents[agent_offset + index].win_count += 1

            winning_agents = sorted(agents, key=lambda x: x.win_count, reverse=True)[:select_number]  # choose the best ones to keep and repopulate
            agents = winning_agents.copy()
            logger.info('Top 4 win rates:')
            for i, each_agent in enumerate(winning_agents[:4]):
                logger.info(f'\tAgent #{i+1}', win_rate=each_agent.win_count / games_per_gen)

            if gen_num % 20 == 0:
                most_wins = winning_agents[0].win_count
                best_agent = winning_agents[0]
                for agent in winning_agents:
                    if agent.win_count > most_wins:
                        most_wins = agent.win_count
                        best_agent = agent
                with open(f'{output_folder}/stats_checkpoint_{gen_num}.txt', 'w+') as f:
                    f.write(f'WIN_RATE: {best_agent.win_count} / {num_validation_games}')
                with open(f'{output_folder}/bid_weights_checkpoint_{gen_num}', 'wb') as f:
                    np.save(f, best_agent.bid_weights)
                with open(f'{output_folder}/play_weights_checkpoint_{gen_num}', 'wb') as f:
                    np.save(f, best_agent.play_weights)

            # perturb winner weights to repopulate
            index = 0
            while len(agents) < population_size:
                if rng.random() < mutate_threshold:
                    new_bid_weights = rng.random((1, Bid.BID_LEN))
                    new_play_weights = rng.random((1, Card.CARD_LEN))
                else:
                    new_bid_weights = perturb_mult * (rng.random((1, Bid.BID_LEN)) - 0.5) + winning_agents[index].bid_weights
                    new_play_weights = perturb_mult * (rng.random((1, Card.CARD_LEN)) - 0.5) + winning_agents[index].play_weights
                    # normalize values to [0, 1]
                    bid_min = np.min(new_bid_weights)
                    bid_max = np.max(new_bid_weights)
                    new_bid_weights = (new_bid_weights - bid_min) / (bid_max - bid_min)
                    play_min = np.min(new_play_weights)
                    play_max = np.max(new_play_weights)
                    new_play_weights = (new_play_weights - play_min) / (play_max - play_min)

                #! switch the below two lines to toggle bid weight optimization
                agents.append(cls(play_weights=new_play_weights))
                # agents.append(cls(bid_weights=new_bid_weights, play_weights=new_play_weights))
                winning_agents[index].win_count = 0  # reset while we're going through the winning agents anyway
                index = (index + 1) % len(winning_agents)

            winning_agents.clear()

        # after final evolution, run a number of games and output the weights with the highest win rate
        for gen_num in range(num_validation_games):
            print(f'Validation game {gen_num}')
            rng.shuffle(agents)
            mp_queue = multiprocessing.Queue()
            jobs = []
            for game_num in range(population_size // 4):
                agent_offset = game_num * 4
                players = agents[agent_offset:agent_offset + 4]
                #! Uncomment below and comment process stuff to disable multiprocessing
                # spades_game = Spades(players, max_rounds=20)
                # result = spades_game.game()
                # result['pid'] = agent_offset
                # queue.put(result)
                process = multiprocessing.Process(target=multiprocess_spades_game, args=(mp_queue, agent_offset, players), kwargs=dict(max_rounds=max_rounds))
                process.start()
                jobs.append(process)

                # wait for core_count processes at a time since only that many can run simultaneously
                if len(jobs) == core_count or len(jobs) == population_size // 4 - 1:
                    for process in jobs:
                        process.join()
                    jobs.clear()

            while not mp_queue.empty():
                results = mp_queue.get()
                agent_offset = results.get('pid')
                if results.get('winning_players') is not None:
                    for index in results.get('winning_players'):
                        agents[agent_offset + index].win_count += 1

        most_wins = agents[0].win_count
        best_agent = agents[0]
        for agent in agents:
            if agent.win_count > most_wins:
                most_wins = agent.win_count
                best_agent = agent

        print(f'Best agent had a win rate of {best_agent.win_count}/{num_validation_games}')

        with open(f'{output_folder}/stats.txt', 'w+') as f:
            f.write(f'WIN_RATE: {best_agent.win_count} / {num_validation_games}')
        with open(f'{output_folder}/bid_weights_final', 'wb') as f:
            np.save(f, best_agent.bid_weights)
        with open(f'{output_folder}/play_weights_final', 'wb') as f:
            np.save(f, best_agent.play_weights)

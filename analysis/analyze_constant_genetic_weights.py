import multiprocessing
import os, sys
import fire
import numpy as np
import matplotlib.pyplot as plt

p = os.path.abspath('.')
sys.path.append(p)
from util import logger
from cards import Card
from spades import Spades, multiprocess_spades_game
from agent import DummyAgent, UserAgent, GreedyAgent
from ai_agents.genetic import ConstantWeightsGenetic


def play_n_games(players, num_games, max_rounds, core_count=4):
    exceeded_rounds = 0
    team_0_wins = 0
    team_1_wins = 0
    for _ in range(num_games):
        spades_game = Spades(players)
        results = spades_game.game()
        if results.get('winning_players') is not None:
            if results.get('winning_players')[0] == 1:
                team_1_wins += 1
            else:
                team_0_wins += 1
        else:
            exceeded_rounds += 1
    
    return exceeded_rounds, team_0_wins, team_1_wins


def cwg_vs_dummy(bid_weights, play_weights, num_games, max_rounds):
    logger.info('Playing CWG vs Dummy', num_games=num_games)
    players = [DummyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights),
               DummyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights)]
    exceeded_rounds, dummy_wins, cwg_wins = play_n_games(players, num_games, max_rounds)

    return (['CWG wins', 'Dummy wins', 'Incomplete games'], [cwg_wins, dummy_wins, exceeded_rounds])


def cwg_vs_greedy(bid_weights, play_weights, num_games, max_rounds):
    logger.info('Playing CWG vs Greedy', num_games=num_games)
    players = [GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights),
               GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights)]
    exceeded_rounds, greedy_wins, cwg_wins = play_n_games(players, num_games, max_rounds)

    return (['CWG wins', 'Greedy wins', 'Incomplete games'], [cwg_wins, greedy_wins, exceeded_rounds])


def learning_timeline(output_folder, num_games, max_rounds, gen_step=20):
    logger.info('Calculating learning timeline CWG vs Greedy')
    gen = 0
    cwg_win_rates = []
    while os.path.exists(f'{output_folder}/bid_weights_checkpoint_{gen}') and os.path.exists(f'{output_folder}/play_weights_checkpoint_{gen}'):
        logger.info(f'Running learning timeline', checkpoint=gen)

        bid_weights = np.load(f'{output_folder}/bid_weights_checkpoint_{gen}')
        play_weights = np.load(f'{output_folder}/play_weights_checkpoint_{gen}')
        players = [GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights),
                   GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights)]

        exceeded_rounds, greedy_wins, cwg_wins = play_n_games(players, num_games, max_rounds)
        cwg_win_rates.append(cwg_wins / (cwg_wins + greedy_wins))
        gen += gen_step
    
    # now add the final results on
    bid_weights = np.load(f'{output_folder}/bid_weights_final')
    play_weights = np.load(f'{output_folder}/play_weights_final')
    players = [GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights),
               GreedyAgent(), ConstantWeightsGenetic(bid_weights=bid_weights, play_weights=play_weights)]

    exceeded_rounds, greedy_wins, cwg_wins = play_n_games(players, num_games, max_rounds)
    cwg_win_rates.append(cwg_wins / (cwg_wins + greedy_wins))

    return (np.arange(0, gen + 1, gen_step), cwg_win_rates)



def main(output_folder=None, bid_weights=None, play_weights=None, num_games=100, timeline=False, max_rounds=100):
    if output_folder is None:
        if bid_weights is None or play_weights is None:
            print('Either Output folder path or Bid and Play weight filepaths must be provided')
            sys.exit()
    else:
        if bid_weights is None:
            bid_weights = output_folder + '/bid_weights_final'
        if play_weights is None:
            play_weights = output_folder + '/play_weights_final'

    bid_weights = np.load(bid_weights)
    play_weights = np.load(play_weights)

    plt.figure()
    xticks = np.arange(0, 14)
    plt.bar(xticks, bid_weights.reshape(-1))
    plt.title('Bid weight preferences')
    plt.xlabel('Bid value')
    plt.ylabel('Relative preferences')

    plt.figure()
    xticks = np.arange(0, 13)
    xnotes = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    yticks = np.arange(0, 4)
    ynotes = ['C', 'D', 'H', 'S']
    plt.imshow(play_weights.reshape((4, 13)))
    plt.title('Card weight preferences')
    plt.xlabel('Card value')
    plt.ylabel('Suit')
    plt.xticks(xticks, xnotes)
    plt.yticks(yticks, ynotes)

    #! uncomment below to enable dummy agent comparison
    # plt.figure()
    # labels, win_counts = cwg_vs_dummy(bid_weights, play_weights, num_gens, max_rounds)
    # plt.bar(labels, win_counts)
    # plt.title('Constant Weight Genetic vs Dummy (100 games)')
    # plt.xlabel('Game result')
    # plt.ylabel('Count')

    plt.figure()
    labels, win_counts = cwg_vs_greedy(bid_weights, play_weights, num_games, max_rounds)
    plt.bar(labels, win_counts)
    plt.title('Constant Weight Genetic vs Greedy (100 games)')
    plt.xlabel('Game result')
    plt.ylabel('Count')

    if timeline:
        plt.figure()
        gens, cwg_win_rates = learning_timeline(output_folder, num_games, max_rounds)
        best_cwg = np.argmax(cwg_win_rates)
        num_gens = len(gens)
        plt.annotate(f'Best Generation: {gens[best_cwg]}\nWin Rate: {cwg_win_rates[best_cwg]}', (gens[best_cwg], cwg_win_rates[best_cwg]),
                     xytext=(gens[-1], np.min(cwg_win_rates)), horizontalalignment='right')
        plt.plot(gens, cwg_win_rates, 'o-')
        plt.title('Constant Weight Genetic vs Greedy Win Rate Over Time')
        plt.xlabel('Generation number')
        plt.ylabel('CWG Win Rate')
        plt.savefig(f'{output_folder}/learning_timeline')

    plt.show()


if __name__ == '__main__':
    fire.Fire(main)

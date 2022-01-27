import cProfile
import pstats
from fire import Fire

from spades import Spades
from agent import DummyAgent, UserAgent
from ai_agents.genetic import ConstantWeightsGenetic


def genetic_training(experiment_name):
    """
    population_size=128, select_number=16, num_generations=1000, games_per_gen=20, num_validation_games=100
    mutate_threshold=0.1, perturb_mult=0.1, output_folder=f'output_{experiment_name}
    """
    ConstantWeightsGenetic.train(population_size=128, select_number=16, num_generations=1000, games_per_gen=20, num_validation_games=100,
                                 mutate_threshold=0.1, perturb_mult=0.1, output_folder=f'output_{experiment_name}')


def main(name='', profile=False):
    if profile:
        with cProfile.Profile() as pr:
            genetic_training(name)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
        stats.dump_stats(filename='profile.prof')
    else:
        genetic_training(name)


if __name__ == '__main__':
    Fire(main)

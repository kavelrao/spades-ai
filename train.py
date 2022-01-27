import cProfile
import pstats
from fire import Fire

from spades import Spades
from agent import DummyAgent, UserAgent
from ai_agents.genetic import ConstantWeightsGenetic


def genetic_training():
    ConstantWeightsGenetic.train(population_size=64, select_number=16, num_generations=100, games_per_gen=10,
                                 num_validation_games=100)


def main(profile=False):
    if profile:
        with cProfile.Profile() as pr:
            genetic_training()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
        stats.dump_stats(filename='profile.prof')
    else:
        genetic_training()


if __name__ == '__main__':
    Fire(main)

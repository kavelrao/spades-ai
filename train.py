import cProfile
import pstats
import os
from fire import Fire

from ai_agents.genetic import ConstantWeightsGenetic


def genetic_training(experiment_name, **kwargs):
    ConstantWeightsGenetic.train(output_folder=f'output_{experiment_name}', **kwargs)


def main(name, profile=False, debug=False, **kwargs):
    if debug:
        os.environ.update(DEBUG="1")
    else:
        os.environ.pop('DEBUG', None)

    if profile:
        with cProfile.Profile() as pr:
            genetic_training(name, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
        stats.dump_stats(filename='profile.prof')
    else:
        genetic_training(name, **kwargs)


if __name__ == '__main__':
    Fire(main)

from spades import Spades
from agent import DummyAgent, UserAgent
from ai_agents.genetic import ConstantWeightsGenetic


def genetic_training():
    ConstantWeightsGenetic.train(population_size=52, num_generations=100, games_per_gen=10, num_validation_games=100)


def main():
    genetic_training()


if __name__ == '__main__':
    main()

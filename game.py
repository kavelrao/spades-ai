from ai_agents.genetic import ConstantWeightsGenetic
from spades import Spades
from agent import DummyAgent, UserAgent


def dummy_game():
    players = []
    for _ in range(4):
        players.append(DummyAgent())
    return players


def one_user_game():
    players = [DummyAgent(), UserAgent(), DummyAgent(), DummyAgent()]
    return players


def one_genetic_game():
    players = [UserAgent(), ConstantWeightsGenetic(bid_weights_file='output/constant_weights_genetic__bid_weights', play_weights_file='output/constant_weights_genetic__play_weights'), UserAgent(), 
    ConstantWeightsGenetic(bid_weights_file='output/constant_weights_genetic__bid_weights', play_weights_file='output/constant_weights_genetic__play_weights')]
    return players


def main():
    spades_game = Spades(one_genetic_game())
    spades_game.game()


if __name__ == '__main__':
    main()

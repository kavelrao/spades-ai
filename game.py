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


def main():
    spades_game = Spades(dummy_game())
    spades_game.game()


if __name__ == '__main__':
    main()

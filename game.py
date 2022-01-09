from spades import Spades
from agent import DummyAgent


def main():
    players = []
    for _ in range(4):
        players.append(DummyAgent())
    spades_game = Spades(players)
    spades_game.game()


if __name__ == '__main__':
    main()

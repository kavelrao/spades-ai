import os, sys
import fire
import numpy as np
import matplotlib.pyplot as plt

p = os.path.abspath('.')
sys.path.append(p)
from cards import Card


def main(bid_weight_file=None, play_weight_file=None):
    if bid_weight_file is None or play_weight_file is None:
        print('Bid and Play weight files must be provided')
        sys.exit()
    bid_weights = np.load(bid_weight_file)
    play_weights = np.load(play_weight_file)

    plt.figure()
    xticks = np.arange(0, 14)
    plt.bar(xticks, bid_weights.reshape(-1))
    plt.title('Bid weight preferences')
    plt.xlabel('Bid value')
    plt.ylabel('Relative preferences')

    plt.figure()
    xticks = np.arange(0, 52)
    xcards = [str(Card(i)) for i in range(len(xticks))]
    plt.bar(xticks, play_weights.reshape(-1))
    plt.title('Card weight preferences')
    plt.xlabel('Card value')
    plt.ylabel('Relative preferences')
    plt.xticks(xticks, xcards, rotation=90)

    plt.show()


if __name__ == '__main__':
    fire.Fire(main)

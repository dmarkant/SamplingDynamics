import matplotlib.pyplot as plt
import numpy as np


def plot_game_with_model(pars, result):
    fig, ax = plt.subplots()
    for i in range(len(pars['data']['sampledata'])):
        opt = pars['data']['sampledata'][i]
        label = ['blue', 'red'][opt]
        ax.plot(i, 1.1, 'o', color=label, alpha=.5)
    ax.plot(i+1, 1.1, 'o', color='black', alpha=.5)
    ax.plot(result['p_stop'], label='p(stop)', color='black')
    ax.plot(result['p_sample_A'], label='p(sample A)', color='blue')
    ax.plot(result['p_sample_B'], label='p(sample B)', color='red')
    ax.set_ylim(-.1, 1.2)
    ax.set_xlim(-1, len(pars['data']['sampledata']) + 1)
    ax.legend(loc='lower left')
    plt.show()

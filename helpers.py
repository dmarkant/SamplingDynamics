import matplotlib.pyplot as plt
import numpy as np


def plot_game_with_model(pars, result):
    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(len(pars['data']['samples'])):
        opt = pars['data']['samples'][i]
        label = ['blue', 'red'][opt]
        ax.plot(i, 1.1, 'o', color=label, alpha=.5)
        ax.text(i, 1.2, pars['data']['outcomes'][i])

    ax.plot(i+1, 1.1, 'o', color='black', alpha=.5)
    ax.plot(result['p_stop'], label='p(stop)', color='black')
    ax.plot(result['p_sample_A'], label='p(sample A)', color='blue')
    ax.plot(result['p_sample_B'], label='p(sample B)', color='red')
    ax.set_ylim(-.1, 1.4)
    ax.set_xlim(-1, len(pars['data']['samples']) + 1)
    ax.legend(loc='lower left')
    plt.show()

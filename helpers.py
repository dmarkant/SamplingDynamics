import matplotlib.pyplot as plt
import numpy as np


def plot_game_with_model(pars, result, ax=None):
    if ax is None:
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

    if ax is None:
        plt.show()
    else:
        return ax


def plot_preference_state(pref, choice, theta, sd, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot([0, len(pref)], [ theta,  theta], '-', color='blue')
    ax.fill_between([0, len(pref)], theta-sd, theta+sd, facecolor='blue', alpha=.1)

    ax.plot([0, len(pref)], [-theta, -theta], '-', color='red')
    ax.fill_between([0, len(pref)], -theta-sd, -theta+sd, facecolor='red', alpha=.1)

    ax.plot(pref, '-o', color='blue' if choice==0 else 'red')

    if ax is None:
        plt.show()
    else:
        return ax

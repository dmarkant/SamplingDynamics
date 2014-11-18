import matplotlib.pylab as plt
import numpy as np
from cogmod.cpt import pweight_prelec, util

"""
try to use core bic in cogmod.fitting
def bic(f, pars):
    return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['samplesize'].size for d in pars['data']]))
"""


def add_problem_labels(ax, label_h, ylim):
    ax.plot([21, 21], ylim, 'k--')
    ax.plot([42, 42], ylim, 'k--')
    ax.plot([63, 63], ylim, 'k--')
    t = ax.text(10, label_h, "Gain, HH-LL", ha="center", va="center", size=15)
    t = ax.text(31, label_h, "Gain, HL-LH", ha="center", va="center", size=15)
    t = ax.text(52, label_h, "Loss, HH-LL", ha="center", va="center", size=15)
    t = ax.text(74, label_h, "Loss, HL-LH", ha="center", va="center", size=15)
    return ax


def plot_model(results):
    #V = np.arange(0, result['states_t'].shape[1]) - result['states_t'].shape[1]/2
    fig, ax = plt.subplots(3, 1, figsize=(6,10))
    ax[0].matshow(results['states_t'].transpose())
    #ax[0].set_yticks(range(0, len(V), 2))
    #ax[0].set_yticklabels(V)
    ax[0].set_ylabel('state')
    ax[0].set_xlabel('samples')

    ax[1].plot(results['resp_prob_t'][:,0], label='L', color='blue')
    ax[1].plot(results['resp_prob_t'][:,1], label='H', color='red')
    ax[1].set_ylabel('response prob')
    ax[1].set_xlabel('samples')
    ax[1].legend()

    ax[2].plot(results['p_stop_t'])
    ax[2].set_ylabel('p(stop|resp)')
    ax[2].set_xlabel('samples')

    plt.show()


def plot_probability_weighting_fcn_prelec(elevation, gamma):
    ps = np.arange(0, 1, .01)
    fig, ax = plt.subplots()
    ax.plot(ps, ps, '--')
    ax.plot(ps, [pweight_prelec(p, elevation, gamma) for p in ps], label='model')
    ax.set_aspect(1)
    ax.set_xlabel('true')
    ax.set_ylabel('weighted')
    ax.set_title('Probability weighting fcn')
    ax.legend()
    plt.show()


def plot_value_weighting_fcn(pow_gain, pow_loss=1, w_loss=1):
    xs = np.arange(-30, 30, .05)
    fig, ax = plt.subplots()
    ax.plot(xs, xs, '--')
    ax.plot(xs, [util(x, pow_gain, pow_loss, w_loss) for x in xs], label='best-fitting pow_gain, pow_loss')
    ax.set_aspect(1)
    ax.set_xlabel('true')
    ax.set_ylabel('weighted')
    ax.set_title('Value weighting fcn')
    ax.set_ylim([-20,20])
    ax.legend()
    plt.show()


def expected_value(option):
    A, B, p_A = option[:3]
    return p_A * A + (1 - p_A) * B



def plot_gamble(g, xlim=[-100, 100]):

    sc = 25

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot([0, 0], [-2, 2], linestyle='--', c='gray')

    ax.plot([g['H'][0]], [1], marker='o', markersize=g['H'][2]*sc, markeredgewidth=0, linestyle='-', color='red', alpha=.4)
    ax.plot([g['H'][1]], [1], marker='o', markersize=(1-g['H'][2])*sc, markeredgewidth=0, linestyle='-', color='red', alpha=.4)

    ax.plot([g['L'][0]], [-1], marker='o', markersize=g['L'][2]*sc, markeredgewidth=0, linestyle='-', color='blue', alpha=.4)
    ax.plot([g['L'][1]], [-1], marker='o', markersize=(1-g['L'][2])*sc, markeredgewidth=0, linestyle='-', color='blue', alpha=.4)



    ax.plot([g['H'][0], g['H'][1]], [1, 1], markeredgewidth=0, linestyle='-', color='red', alpha=.2)
    ax.plot([g['L'][0], g['L'][1]], [-1, -1], markeredgewidth=0, linestyle='-', color='blue', alpha=.2)

    ax.plot([expected_value(g['H'])], [1], marker='x', markersize=14, color='red')
    ax.plot([expected_value(g['L'])], [-1], marker='x', markersize=14, color='blue')
    ax.set_xlim(xlim)
    ax.set_ylim([-2, 2])
    ax.set_xlabel('Outcome')
    ax.axes.get_yaxis().set_visible(False)
    return fig, ax


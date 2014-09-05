import matplotlib.pylab as plt
import numpy as np


def bic(f, pars):
    return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['samplesize'].size for d in pars['data']]))


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
    fig, ax = plt.subplots(1, 3, figsize=(20,4))
    ax[0].matshow(results['states_t'].transpose())
    #ax[0].set_yticks(range(0, len(V), 2))
    #ax[0].set_yticklabels(V)
    ax[0].set_ylabel('state')
    ax[0].set_xlabel('time')

    ax[1].plot(results['resp_prob_t'][:,0], label='A')
    ax[1].plot(results['resp_prob_t'][:,1], label='B')
    ax[1].set_ylabel('response prob')
    ax[1].set_xlabel('time')
    ax[1].legend()

    ax[2].plot(results['p_stop_t'])
    ax[2].set_ylabel('p(stop|resp)')
    ax[2].set_xlabel('time')

    plt.show()



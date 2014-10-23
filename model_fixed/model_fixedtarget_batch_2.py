import numpy as np
from mypy.explib.hau2008 import hau2008
from scipy.optimize import minimize
from fitting import *

import sys
sys.path.append("../")
from switching import count_streaks



def pr_sample(n, t, s):
    """
    Probability of taking another sample.

    n -- Number of samples so far for this option
    t -- Target number of samples
    s -- Scale parameter; as s increases, sharper gain
    """
    d = (n + 1) - t
    p = 1. / (1. + np.exp(s * d))
    return p


def run(pars):

    verbose = pars.get('verbose', False)
    data    = pars.get('data')['samples']

    t = np.round(pars.get('target', 15)) # target total sample size
    s = pars.get('s', 1.)               # continue scale factor

    t_batch = np.round(pars.get('target_batch', 5)) # target total sample size
    s_batch = pars.get('s_batch', 1.)

    count = np.arange(len(data) + 1)
    count_streak = count_streaks(data)


    # probability of switching is based on streak count
    p_stay = 1. / (1. + np.exp((np.array(count_streak) + 1 - t_batch) * s_batch))

    # probability of stopping is based on total count
    p_sample = 1. / (1. + np.exp((count + 1 - t) * s))

    p_sample_A = p_stay[1:] * (data==0) + (1 - p_stay[1:]) * p_sample[1:] * (data==1)
    p_sample_B = p_stay[1:] * (data==1) + (1 - p_stay[1:]) * p_sample[1:] * (data==0)

    p_sample_A = np.concatenate(([0.5], p_sample_A))
    p_sample_B = np.concatenate(([0.5], p_sample_B))

    p_stop = 1 - (p_sample_A + p_sample_B)

    #print np.round(p_sample_A, 3)
    #print np.round(p_sample_B, 3)
    #print p_sample_A + p_sample_B

    #print p_stop + p_sample_A + p_sample_B


    return {'p_stop': p_stop,
            'p_sample_A': p_sample_A,
            'p_sample_B': p_sample_B}


def nloglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)

    llh = 0.
    for trial, option in enumerate(pars['data']['sampledata']):
        if option == 0:
            llh += np.log(pfix(result['p_sample_A'][trial]))
        else:
            llh += np.log(pfix(result['p_sample_B'][trial]))


    llh += np.log(pfix(result['p_stop'][-1]))

    nllh = -1 * llh
    return nllh


def nloglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    alldata = pars['data']

    nllh = []
    for data in alldata:
        _pars = deepcopy(pars)
        _pars['data'] = data
        result = run(_pars)

        llh = 0.
        for trial, option in enumerate(_pars['data']['sampledata']):
            if option == 0:
                llh += np.log(pfix(result['p_sample_A'][trial]))
            else:
                llh += np.log(pfix(result['p_sample_B'][trial]))

        # stop decision
        llh += np.log(pfix(result['p_stop'][-1]))

        nllh.append(-1 * llh)

    return np.sum(nllh)


def fit_subject_across_gambles(data):

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['sampledata'].size + 1 for d in pars['data']]))

    #for d in data:
    #    print d['sampledata']

    # find the highest number of samples for a game
    counts = []
    for d in data:
        counts.append(len(d['sampledata']))
    max_count = np.max(counts)

    tset = []
    for target in range(max_count + 10):
        for target_batch in range(1, target):
            tset.append([target, target_batch])
    print len(tset)

    fitresults = {}
    nllh = []
    for target, target_batch in tset:
        pars = {'data': data,
                'target': target,
                'target_batch': target_batch,
                'fitting': ['s', 's_batch']}
        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        nllh.append(f['fun'])
        fitresults[(target, target_batch)] = f

    bf_t = tset[np.argmin(nllh)]
    fr = fitresults[tuple(bf_t)]

    print 'best fitting target sizes:', bf_t
    print 'bic:', bic(fr, pars)
    print fr

    return {'bf_par': {'target': bf_t[0],
                       's': fr['x'][0],
                       'target_batch': bf_t[1],
                       's_batch': fr['x'][1]},
            'nllh': fr['fun'],
            'bic': bic(fr, pars),
            'success': fr['success']}



if __name__ == '__main__':

    options = hau2008.get_options(1, 0)

    df_samples, df_choices = hau2008.load_study(1)

    subj = 1
    sdata = df_samples[(df_samples['subject']==subj) & (df_samples['problem']==1)]
    sampledata = sdata['option'].values
    choice = df_choices[(df_choices['subject']==subj) & (df_choices['problem']==1)]['choice'].values[0]

    # run model for a single gamble
    pars = {'data': {'sampledata': sampledata,
                     'choice': choice},
            'target': 5,
            's': 1.}

    print run(pars)

    # get likelihood for a single gamble
    pars = {'data': {'sampledata': sampledata,
                     'choice': choice},
            's': 1.,
            'fitting': ['target']}
    print nloglik([10.], pars)


    # get likelihood across all gambles for a single subject
    sdata = df_samples[(df_samples['subject']==subj)]
    problems = np.sort(sdata['problem'].unique())
    data = []
    for problem in problems:
        data.append({'sampledata': sdata[sdata['problem']==problem]['option'].values,
                     'choice': df_choices[(df_choices['subject']==subj) & (df_choices['problem']==problem)]['choice'].values[0]})

    print fit_subject_across_gambles(data)
    """
    pars = {'data': data,
            's': 1.,
            'fitting': ['target']}
    for targ in [5., 10., 15., 20.]:
        print nloglik_across_gambles([targ], pars)


    for d in data:
        print d['sampledata']

    # grid search on target size, fit scaling
    for targ in range(30):
        pars = {'data': data,
                'target': targ,
                'fitting': ['s']}

        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        print targ, f
    """


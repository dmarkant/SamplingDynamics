import numpy as np
from mypy.explib.hau2008 import hau2008
from scipy.optimize import minimize
from fitting import *




def pr_sample(n, t, s):
    """
    Probability of taking another sample from the
    same option.

    n -- Number of samples so far for this option
    t -- Target number of samples
    s -- Scale parameter; as s increases, sharper gain
    """
    d = (n + 1) - t
    p = 1. / (1. + np.exp(s * d))
    return p


def run(pars):

    verbose = pars.get('verbose', False)
    data    = pars.get('data')['sampledata']

    t = np.round(pars.get('target', 5)) # target sample size per option
    s = pars.get('s', 1.)               # continue scale factor

    counts = [0, 0] # assuming 2 options
    p_sample = []

    for trial, option in enumerate(data):

        # did a switch occur?
        if trial==0 or option==data[trial-1]:
            switch = False
        else:
            switch = True

        if not switch:
            p = pr_sample(counts[option], t, s)
        else:
            p = (1 - pr_sample(counts[data[trial-1]], t, s)) * pr_sample(counts[option], t, s)
        p_sample.append(p)

        counts[option] += 1

    # get stopping probability
    p_stop = (1 - pr_sample(counts[0], t, s)) * (1 - pr_sample(counts[1], t, s))

    return {'p_sample': p_sample,
            'p_stop': p_stop}


def nloglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)
    nllh = -1 * (np.sum(np.log(result['p_sample'])) + np.log(result['p_stop']))
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
        nllh.append(-1 * (np.sum(np.log(result['p_sample'])) + np.log(result['p_stop'])))

    return np.sum(nllh)


def fit_subject_across_gambles(data):

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['sampledata'].size for d in pars['data']]))

    for d in data:
        print d['sampledata']

    # find the highest number of samples for a single option
    counts = []
    for d in data:
        counts.append(np.sum(d['sampledata']==0))
        counts.append(np.sum(d['sampledata']==1))
    max_count = np.max(counts)

    # grid search on target size, fit scaling
    fitresults = {}
    nllh = []
    for targ in range(max_count * 2):
        pars = {'data': data,
                'target': targ,
                'fitting': ['s']}

        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        nllh.append(f['fun'])
        fitresults[targ] = f

    bf_targ =  np.argmin(nllh)
    print 'best fitting target size:', bf_targ
    print 'bic:', bic(fitresults[bf_targ], pars)
    print fitresults[bf_targ]

    return {'bf_par': {'target': bf_targ,
                       's': fitresults[bf_targ]['x'][0]},
            'nllh': fitresults[bf_targ]['fun'],
            'bic': bic(fitresults[bf_targ], pars),
            'success': fitresults[bf_targ]['success']}



if __name__ == '__main__':

    options = hau2008.get_options(1, 0)

    df_samples, df_choices = hau2008.load_study(1)

    subj = 8
    sdata = df_samples[(df_samples['subject']==subj) & (df_samples['problem']==1)]
    sampledata = sdata['option'].values

    # run model for a single gamble
    pars = {'data': {'sampledata': sampledata},
            'target': 5,
            's': 1.}

    print run(pars)

    # get likelihood for a single gamble
    pars = {'data': {'sampledata': sampledata},
            's': 1.,
            'fitting': ['target']}
    print nloglik([10.], pars)


    # get likelihood across all gambles for a single subject
    sdata = df_samples[(df_samples['subject']==subj)]
    problems = np.sort(sdata['problem'].unique())
    data = []
    for problem in problems:
        data.append({'sampledata': sdata[sdata['problem']==problem]['option'].values})

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


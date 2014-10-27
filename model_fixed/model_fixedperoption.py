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
    data    = pars.get('data')['samples']

    t = np.round(pars.get('target', 5)) # target sample size per option
    s = pars.get('s', 1.)               # continue scale factor

    counts_A = np.cumsum((data==0))
    counts_B = np.cumsum((data==1))

    p_sample_A = 1. / (1. + np.exp((counts_A + 1 - t) * s))
    p_sample_B = 1. / (1. + np.exp((counts_B + 1 - t) * s))

    p_sample_A = p_sample_A * (data==0) + (1 - p_sample_B) * p_sample_A * (data==1)
    p_sample_B = p_sample_B * (data==1) + (1 - p_sample_A) * p_sample_B * (data==0)

    p_sample_A = np.concatenate(([0.5], p_sample_A))
    p_sample_B = np.concatenate(([0.5], p_sample_B))

    p_stop = 1 - (p_sample_A + p_sample_B)

    return {'p_stop': p_stop,
            'p_sample_A': p_sample_A,
            'p_sample_B': p_sample_B}


def nloglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)

    llh = 0.
    for trial, option in enumerate(pars['data']['samples']):
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
        for trial, option in enumerate(_pars['data']['samples']):
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
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['samples'].size + 1 for d in pars['data']]))

    #for d in data:
    #    print d['sampledata']

    # find the highest number of samples for a single option
    counts = []
    for d in data:
        counts.append(np.sum(d['samples']==0))
        counts.append(np.sum(d['samples']==1))
    max_count = np.max(counts)

    # grid search on target size, fit scaling
    fitresults = {}
    nllh = []
    targets = range(max_count * 2)
    for targ in targets:
        pars = {'data': data,
                'target': targ,
                'fitting': ['s']}

        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        nllh.append(f['fun'])
        #print '(grid) target size: %i, nllh: %s' % (targ, f['fun'])
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

    subj = 1
    sdata = df_samples[(df_samples['subject']==subj) & (df_samples['problem']==1)]
    sampledata = sdata['option'].values
    choice = df_choices[(df_choices['subject']==subj) & (df_choices['problem']==1)]['choice'].values[0]

    # run model for a single gamble
    pars = {'data': {'samples': sampledata,
                     'choice': choice},
            'target': 5,
            's': 1.}

    print run(pars)

    # get likelihood for a single gamble
    pars = {'data': {'samples': sampledata,
                     'choice': choice},
            's': 1.,
            'fitting': ['target']}
    print nloglik([10.], pars)


    # get likelihood across all gambles for a single subject
    sdata = df_samples[(df_samples['subject']==subj)]
    problems = np.sort(sdata['problem'].unique())
    data = []
    for problem in problems:
        data.append({'samples': sdata[sdata['problem']==problem]['option'].values,
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


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

    t = np.round(pars.get('target', 5)) # target total sample size
    s = pars.get('s', 1.)               # continue scale factor
    rho = pars.get('rho', 0.5)          # switching probability

    p_sample = 1. / (1. + np.exp((np.arange(1, len(data) + 1) + 1 - t) * s))
    p_stop_m = np.concatenate(([0.], 1 - p_sample))

    p_sample_A_m = p_sample[:-1] * ((1 - rho) * (data[:-1]==0) + rho * (data[:-1]==1))
    p_sample_A_m = np.concatenate(([0.5], p_sample_A_m))

    p_sample_B_m = p_sample[:-1] * ((1 - rho) * (data[:-1]==1) + rho * (data[:-1]==0))
    p_sample_B_m = np.concatenate(([0.5], p_sample_B_m))

    """
    count = 0 # number of samples so far
    p_sample_A = []
    p_sample_B = []
    p_stop = []

    for trial, option in enumerate(data):

        # on the first trial, sample randomly
        if trial==0:
            p_sample_A.append(0.5)
            p_sample_B.append(0.5)
            p_stop.append(0.)
        else:

            pr = pr_sample(count, t, s)
            p_stop.append(1 - pr)

            if data[trial-1] == 0:
                p_sample_A.append(pr * (1 - rho))
                p_sample_B.append(pr * rho)
            else:
                p_sample_A.append(pr * rho)
                p_sample_B.append(pr * (1 - rho))

        count += 1

    # final stopping decision
    p_stop.append(1 - pr_sample(count, t, s))
    """

    return {'p_stop': p_stop_m,
            'p_sample_A': p_sample_A_m,
            'p_sample_B': p_sample_B_m}


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
    counts = [len(d['sampledata']) for d in data]

    # grid search on target size, fit scaling
    fitresults = {}
    nllh = []
    targets = range(np.min(counts), np.max(counts) + 20)
    for targ in targets:
        pars = {'data': data,
                'target': targ,
                'fitting': ['s', 'rho']}

        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        nllh.append(f['fun'])
        #print targ, f['fun']
        fitresults[targ] = f

    bf_targ =  targets[np.argmin(nllh)]
    print 'best fitting target size:', bf_targ
    print 'bic:', bic(fitresults[bf_targ], pars)
    print fitresults[bf_targ]

    return {'bf_par': {'target': bf_targ,
                       's': fitresults[bf_targ]['x'][0],
                       'rho': fitresults[bf_targ]['x'][1]},
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


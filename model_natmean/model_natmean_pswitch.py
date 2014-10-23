from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma, lognorm
from cogmod.cpt import util
from mypy.explib.hau2008 import hau2008
from fitting import *
from model_natmean import get_state_trajectory


def run(pars):

    verbose   = pars.get('verbose', False)
    options   = pars.get('options', None)
    data      = pars.get('data')

    # probability of switching
    rho       = pars.get('rho', .5)

    # sequential natural mean
    eval_crit = pars.get('eval_crit', 0.)
    eval_pow  = pars.get('eval_pow', 1.)

    mu        = pars.get('mu', 10)
    sd        = pars.get('sd', 1)

    #alpha     = pars.get('alpha', 2.)
    #beta      = pars.get('beta', 1.)

    # guessing probability
    p_guess   = pars.get('p_guess', 0.)

    # get preference trajectory
    obs = np.transpose((data['samples'], data['outcomes']))
    pref = get_state_trajectory(data['options'],
                                obs,
                                eval_crit,
                                eval_pow)['states']

    # normal distribution
    p_stop = norm.cdf(np.abs(pref), loc=mu, scale=sd) / (1. - norm.cdf(0, loc=mu, scale=sd))

    # on each trial, the probability of crossing the boundary is
    # determined by the distribution over separation sizes
    #p_stop = gamma.cdf(np.abs(pref), alpha, scale=1./beta)
    p_stop[0] = 0.

    p_samp = 1 - p_stop
    d = np.array(data['samples'])
    p_sample_A = p_samp[1:] * ((1 - rho) * (d==0) + rho * (d==1))
    p_sample_A = np.concatenate(([.5], p_sample_A))

    p_sample_B = p_samp[1:] * ((1 - rho) * (d==1) + rho * (d==0))
    p_sample_B = np.concatenate(([.5], p_sample_B))

    # at end of sampling, give choice probabilities
    if pref[-1] == 0:
        p_choice = [.5, .5]
    elif pref[-1] > 0:
        p_choice = [1 - (p_guess/2.), p_guess/2.]
    else:
        p_choice = [p_guess/2., 1 - p_guess/2.]

    return {'pref': pref,
            'p_stop': p_stop,
            'p_sample_A': p_sample_A,
            'p_sample_B': p_sample_B,
            'p_choice': p_choice}

"""
def nloglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)

    sampledata = pars['data']['sampledata']

    llh_sampling = 0.
    for trial, obs in enumerate(sampledata):

        if obs == 0:
            llh_sampling += np.log(pfix(result['p_sample_A'][trial]))
        else:
            llh_sampling += np.log(pfix(result['p_sample_B'][trial]))

    p_stop = result['p_stop'][-1]
    llh_sampling += np.log(pfix(p_stop))


    # stop/choice
    if pars['data']['choice'] == 0:
        top = result['p_stop_choose_A'][-1]
    else:
        top = result['p_stop_choose_B'][-1]

    bottom = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
    p_choice = top / bottom

    #print 'top:', top
    #print 'bottom:', bottom
    #print 'p(choice):', top / bottom

    llh_choice = np.log(pfix(p_choice))

    if len(fitting) > 0:
        return -1 * (llh_sampling + llh_choice)
    else:
        return {'llh_sampling': -llh_sampling,
                'llh_choice': -llh_choice}
"""


def nloglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    obj = pars.get('obj', None)

    alldata = pars['data']

    llh_sampling = 0.
    llh_choice = 0.
    for data in alldata:
        _pars = deepcopy(pars)
        _pars['data'] = data

        result = run(_pars)

        sampledata = data['samples']
        for trial, obs in enumerate(sampledata):

            if obs == 0:
                llh_sampling += np.log(pfix(result['p_sample_A'][trial]))
            else:
                llh_sampling += np.log(pfix(result['p_sample_B'][trial]))

        p_stop = result['p_stop'][-1]
        llh_sampling += np.log(pfix(p_stop))

        choice = _pars['data']['choice']
        p_choice = result['p_choice'][choice]
        llh_choice += np.log(pfix(p_choice))

    print llh_sampling + llh_choice, value


    if obj is 'both':
        return -1 * (llh_sampling + llh_choice)
    elif obj is 'sampling':
        return -llh_sampling
    elif obj is 'choice':
        return -llh_choice
    else:
        return {'llh_sampling': -llh_sampling,
                'llh_choice': -llh_choice}


def fit_subject_across_gambles(data, fixed={}, fitting=[], obj='both'):

    pars = {'data': data,
            'obj': obj,
            'fitting': fitting}
    for parname in fixed:
        pars[parname] = fixed[parname]

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['samples'].size + 1 for d in pars['data']]))

    succeeded = False
    iter = 0
    while not succeeded and iter < 5:
        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        if f['success']:
            succeeded = True
        iter += 1

    return {'bf_par': {fitting[i]: f['x'][i] for i in range(len(fitting))},
            'nllh': f['fun'],
            'bic': bic(f, pars),
            'success': f['success']}


if __name__ == '__main__':

    # load some data
    options = hau2008.get_options(1, 0)

    df_samples, df_choices = hau2008.load_study(1)

    sdata = df_samples[(df_samples['subject']==1) & (df_samples['problem']==1)]
    sampledata = sdata['option'].values
    outcomes = sdata['outcome'].values

    choicedata = df_choices[(df_choices['subject']==1) & (df_choices['problem']==1)]
    choice = choicedata['choice'].values[0]
    print choice


    pars = {'options': options,
            'data': {'sampledata': sampledata,
                     'outcomes': outcomes,
                     'choice': choice},
            'eval_pow': 1.2,
            'theta': 20}

    print run(pars)


    # evaluate log-likelihood
    pars = {'options': options,
            'data': {'sampledata': sampledata,
                     'outcomes': outcomes,
                     'choice': choice},
            'fitting': ['theta'],
            'eval_pow': 1.2}

    print nloglik([10.], pars)


    # evaluate log-likelihood again, but without free parameters, to get
    # separate scores for sampling and choice
    pars = {'options': options,
            'data': {'sampledata': sampledata,
                     'outcomes': outcomes,
                     'choice': choice},
            'fitting': [],
            'theta': 5,
            'eval_pow': 1.2}

    print nloglik([10.], pars)


from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from cogmod.cpt import util
from mypy.explib.hau2008 import hau2008
from fitting import *


def valuation(obs, eval_crit, eval_pow):
    """
    Valuation rule assuming zero value for non-sampled
    option.
    """
    uA = util(obs[1], eval_pow) if obs[0]==0 else 0.
    uB = util(obs[1], eval_pow) if obs[0]==1 else 0.
    return uA - eval_crit - uB


def run(pars):

    verbose = pars.get('verbose', False)
    options = pars.get('options', None)
    data    = pars.get('data')

    rho       = pars.get('rho', .5) # probability of switching
    eval_crit = pars.get('eval_crit', 0.)
    eval_pow  = pars.get('eval_pow', 1.)
    z_mu      = pars.get('z_mu', 0.)
    z_sd      = pars.get('z_sd', 10.)
    theta     = pars.get('theta')

    # first evalute the trajectory
    samples = [0.]
    for trial, obs in enumerate(data['sampledata']):

        # evaluate the outcome
        samples.append(valuation([obs, data['outcomes'][trial]], eval_crit, eval_pow))

    pref = np.cumsum(samples)

    # on each trial, the probability of being in a state is given by
    # normal distribution, centered on the current preference state
    pref_mu = pref + z_mu
    p_stop_choose_A = np.array([norm.cdf(-theta, loc=p, scale=z_sd) for p in pref_mu])
    p_stop_choose_B = np.array([1. - norm.cdf(theta, loc=p, scale=z_sd) for p in pref_mu])
    p_stop = p_stop_choose_A + p_stop_choose_B

    # sampling probabilities, incorporating fixed probability of switching
    p_sample_A = []
    p_sample_B = []
    for trial, obs in enumerate(data['sampledata']):
        p_samp = 1 - p_stop[trial]
        if trial == 0:
            p_sample_A.append(p_samp * .5)
            p_sample_B.append(p_samp * .5)
        else:
            if data['sampledata'][trial-1] == 0:
                # last trial was A
                p_sample_A.append(p_samp * (1 - rho))
                p_sample_B.append(p_samp * rho)
            else:
                # last trial was B
                p_sample_A.append(p_samp * rho)
                p_sample_B.append(p_samp * (1 - rho))


    return {'pref': pref,
            'p_stop_choose_A': p_stop_choose_A,
            'p_stop_choose_B': p_stop_choose_B,
            'p_sample_A': p_sample_A,
            'p_sample_B': p_sample_B}


def loglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)

    # check if fitting sampling and choices
    if 'ignorechoices' in pars:
        ignorechoices = pars['ignorechoices']
    else:
        ignorechoices = False


    sampledata = pars['data']['sampledata']
    llh = 0.
    for trial, obs in enumerate(sampledata):

        if obs[0] == 0:
            llh += np.log(pfix(result['p_sample_A'][trial]))
        else:
            llh += np.log(pfix(result['p_sample_B'][trial]))


    if ignorechoices:
        p_stop = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
        llh += np.log(pfix(p_stop))
    else:
        # stop/choice
        if data['choice'] == 0:
            llh += np.log(pfix(result['p_stop_choose_A'][-1]))
        else:
            llh += np.log(pfix(result['p_stop_choose_B'][-1]))

    return -llh


def nloglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    # check if fitting sampling and choices
    if 'ignorechoices' in pars:
        ignorechoices = pars['ignorechoices']
    else:
        ignorechoices = False


    alldata = pars['data']

    llh = 0.
    for data in alldata:
        _pars = deepcopy(pars)
        _pars['data'] = data

        result = run(_pars)

        sampledata = data['sampledata']
        for trial, obs in enumerate(sampledata):

            if obs == 0:
                llh += np.log(pfix(result['p_sample_A'][trial]))
            else:
                llh += np.log(pfix(result['p_sample_B'][trial]))

        if ignorechoices:
            p_stop = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
            llh += np.log(pfix(p_stop))
        else:
            # stop/choice
            if data['choice'] == 0:
                llh += np.log(pfix(result['p_stop_choose_A'][-1]))
            else:
                llh += np.log(pfix(result['p_stop_choose_B'][-1]))

    return -llh


def fit_subject_across_gambles(data, fixed={}, fitting=[]):

    pars = {'data': data,
            'fitting': fitting}
    for parname in fixed:
        pars[parname] = fixed[parname]

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['sampledata'].size + 1 for d in pars['data']]))

    init = [randstart(par) for par in pars['fitting']]
    f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')

    return {'bf_par': {fitting[i]: f['x'][i] for i in range(len(fitting))},
            'nllh': f['fun'],
            'bic': bic(f, pars),
            'success': f['success']}


if __name__ == '__main__':

    # load some data
    options = hau2008.get_options(1, 0)

    df_samples, df_choices = hau2008.load_study(1)

    sdata = df_samples[(df_samples['subject']==1) & (df_samples['problem']==1)]
    sampledata = sdata[['option', 'outcome']].values

    choicedata = df_choices[(df_choices['subject']==1) & (df_choices['problem']==1)]
    choice = choicedata['choice'].values[0]
    print choice


    pars = {'options': options,
            'data': {'sampledata': sampledata,
                     'choice': choice},
            'eval_pow': 1.2,
            'theta': 20}

    print run(pars)


    # evaluate log-likelihood
    pars = {'options': options,
            'data': {'sampledata': sampledata,
                     'choice': choice},
            'fitting': ['theta'],
            'eval_pow': 1.2}

    print loglik([10.], pars)

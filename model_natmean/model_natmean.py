"""
Common code for natural mean model, both the stand-alone
choice model and the sequential version.


"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from mypy.explib.hau2008 import hau2008
from fitting import *


def util(outcome, pow_gain, pow_loss=None, w_loss=1.):

    if pow_loss is None: pow_loss = pow_gain
    if outcome >= 0.:
        return (outcome ** pow_gain)
    else:
        return (-w_loss * ((-outcome) ** pow_loss))


def valuation(obs, eval_crit, eval_pow):
    """
    Valuation rule assuming zero value for non-sampled
    option.
    """
    uA = util(obs[1], eval_pow) if obs[0]==0 else 0.
    uB = util(obs[1], eval_pow) if obs[0]==1 else 0.
    return uA - eval_crit - uB



"""
Sequential natural mean.

"""
def get_state_trajectory(options, observations, eval_crit, eval_pow):

    # calculate the pooled variance once
    probs = []
    utils = []
    eu    = []
    evar  = []
    for opt_i, option in enumerate(options):
        probs.append([prob for (outcome, prob) in option])
        utils.append([util(outcome, eval_pow) for (outcome, prob) in option])
        eu.append(np.array([probs[opt_i][i] * utils[opt_i][i] for i in range(len(probs[opt_i]))]))
        evar.append(np.sum([probs[opt_i][i] * (utils[opt_i][i] ** 2) for i in range(len(probs[opt_i]))]) - np.sum(eu[opt_i]) ** 2)

    outs      = np.outer(utils[0] - np.sum(eu[0]), utils[1] - np.sum(eu[1]))
    ps        = np.outer(probs[0], probs[1])
    cov       = np.sum(np.multiply(outs, ps))
    pooledvar = np.sum(evar) - 2 * cov

    #pooledvar = 1.

    # assume unbiased starting position
    V = [0.]
    for obs in observations:
        V.append(valuation(obs, eval_crit, eval_pow) / np.sqrt(pooledvar))
    states = np.cumsum(V)

    return {'values': V,
            'states': states}


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
    s         = pars.get('s', 1.)
    p_err     = pars.get('p_err', 0.)

    # first evalute the trajectory
    samples = [0.]
    for trial, obs in enumerate(data['sampledata']):

        # evaluate the outcome
        samples.append(valuation([obs, data['outcomes'][trial]], eval_crit, eval_pow))

    pref = np.cumsum(samples)

    # on each trial, the probability of being in a state is given by
    # normal distribution, centered on the current preference state
    pref_mu = pref + z_mu

    #p_stop_choose_A = [p_stop_choose(p, theta, s) for p in pref_mu]
    #p_stop_choose_B = [p_stop_choose(p, -theta, s) for p in pref_mu]

    p_stop_choose_A = np.array([1. - norm.cdf(theta, loc=p, scale=z_sd) for p in pref_mu])
    p_stop_choose_B = np.array([norm.cdf(-theta, loc=p, scale=z_sd) for p in pref_mu])
    p_stop = p_stop_choose_A + p_stop_choose_B

    print p_stop_choose_A

    p_samp = 1 - p_stop
    d = np.array(data['sampledata'][:-1])
    p_sample_A_m = p_samp[1:-1] * ((1 - rho) * (d==0) + rho * (d==1))
    p_sample_A_m = np.concatenate(([p_samp[0] * .5], p_sample_A_m))

    p_sample_B_m = p_samp[1:-1] * ((1 - rho) * (d==1) + rho * (d==0))
    p_sample_B_m = np.concatenate(([p_samp[0] * .5], p_sample_B_m))


    return {'pref': pref,
            'p_stop_choose_A': p_stop_choose_A,
            'p_stop_choose_B': p_stop_choose_B,
            'p_sample_A': p_sample_A_m,
            'p_sample_B': p_sample_B_m}


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

    p_stop = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
    llh_sampling += np.log(pfix(p_stop))


    # stop/choice
    if pars['data']['choice'] == 0:
        top = result['p_stop_choose_A'][-1]
    else:
        top = result['p_stop_choose_B'][-1]

    bottom = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
    p_choice = top / bottom

    llh_choice = np.log(pfix(p_choice))

    if len(fitting) > 0:
        return -1 * (llh_sampling + llh_choice)
    else:
        return {'llh_sampling': -llh_sampling,
                'llh_choice': -llh_choice}


def nloglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    alldata = pars['data']

    llh_sampling = 0.
    llh_choice = 0.
    for data in alldata:
        _pars = deepcopy(pars)
        _pars['data'] = data

        result = run(_pars)

        sampledata = data['sampledata']
        for trial, obs in enumerate(sampledata):

            if obs == 0:
                llh_sampling += np.log(pfix(result['p_sample_A'][trial]))
            else:
                llh_sampling += np.log(pfix(result['p_sample_B'][trial]))

        p_stop = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
        llh_sampling += np.log(pfix(p_stop))

        if _pars['data']['choice']==0:
            top = result['p_stop_choose_A'][-1]
        else:
            top = result['p_stop_choose_B'][-1]
        bottom = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
        p_choice = top / bottom
        llh_choice += np.log(pfix(p_choice))


    if len(fitting) > 0:
        return -1 * (llh_sampling + llh_choice)
    else:
        return {'llh_sampling': -llh_sampling,
                'llh_choice': -llh_choice}


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


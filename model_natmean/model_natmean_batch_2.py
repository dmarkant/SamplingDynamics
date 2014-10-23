from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma
from cogmod.cpt import util
from mypy.explib.hau2008 import hau2008
from fitting import *
from model_natmean import get_state_trajectory

import sys
sys.path.append("../")
from switching import count_streaks


def valuation(obs, eval_crit, eval_pow):
    """
    Valuation rule assuming zero value for non-sampled
    option.
    """
    uA = util(obs[1], eval_pow) if obs[0]==0 else 0.
    uB = util(obs[1], eval_pow) if obs[0]==1 else 0.
    return uA - eval_crit - uB


def run(pars):

    verbose   = pars.get('verbose', False)
    options   = pars.get('options', None)
    data      = pars.get('data')

    eval_crit = pars.get('eval_crit', 0.)
    eval_pow  = pars.get('eval_pow', .01)

    mu        = pars.get('mu', 10)
    sd        = pars.get('sd', 1)

    # batch sampling
    t_batch = np.round(pars.get('target_batch', 5)) # target total sample size
    s_batch = pars.get('s_batch', 1.)

    # guessing probability
    p_guess   = pars.get('p_guess', 0.)


    # get preference trajectory
    obs = np.transpose((data['samples'], data['outcomes']))
    pref = get_state_trajectory(data['options'],
                                obs,
                                eval_crit,
                                eval_pow)['states']



    # probability of switching is based on streak count
    count_streak = count_streaks(data['samples'])
    p_stay = 1. / (1. + np.exp((np.array(count_streak) + 1 - t_batch) * s_batch))

    # normal distribution
    p_stop = norm.cdf(np.abs(pref), loc=mu, scale=sd) / (1. - norm.cdf(0, loc=mu, scale=sd))
    p_stop[0] = 0.

    # on each trial, the probability of crossing the boundary is
    # determined by the distribution over separation sizes
    #p_stop = gamma.cdf(np.abs(pref), 2., loc=th_shape, scale=th_scale)
    #p_stop[0] = 0.
    p_samp = 1 - p_stop

    d = np.array(data['samples'])

    p_sample_A = p_stay[1:] * (d==0) + (1 - p_stay[1:]) * p_samp[1:] * (d==1)
    p_sample_A = np.concatenate(([.5], p_sample_A))

    p_sample_B = p_stay[1:] * (d==1) + (1 - p_stay[1:]) * p_samp[1:] * (d==0)
    p_sample_B = np.concatenate(([.5], p_sample_B))

    # conditional probability of stopping
    p_stop = 1 - (p_sample_A + p_sample_B)

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


def nloglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    result = run(pars)

    sampledata = pars['data']['samples']

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

    #print 'top:', top
    #print 'bottom:', bottom
    #print 'p(choice):', top / bottom

    llh_choice = np.log(pfix(p_choice))

    if len(fitting) > 0:
        return -1 * (llh_sampling + llh_choice)
    else:
        return {'llh_sampling': -llh_sampling,
                'llh_choice': -llh_choice}


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

    #print '\t%s: %s' % (map(lambda v: np.round(v, 3), value), llh_sampling)

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

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['samples'].size + 1 for d in pars['data']]))

    counts = []
    for d in data:
        counts.append(len(d['samples']))
    max_count = np.max(counts)
    print max_count

    tofit = [p for p in fitting if p!='target_batch']
    fitresults = {}
    nllh = []
    for target_batch in range(max_count):
        pars = {'data': data,
                'obj': obj,
                'target_batch': target_batch,
                'fitting': tofit}
        nllh_tb = []
        fitresults_tb = {}
        for iter in range(5):
            init = [randstart(par) for par in pars['fitting']]
            f = minimize(nloglik_across_gambles, init, (pars,), method='Powell', tol=.00001)
            if f['success'] is True:
                nllh_tb.append(f['fun'])
                fitresults_tb[iter] = f
                print 'success', nllh_tb[-1], np.round(f['x'], 2)
            else:
                print 'failed'
                nllh_tb.append(np.inf)

        best_iter = np.argmin(nllh_tb)
        nllh.append(nllh_tb[best_iter])
        fitresults[target_batch] = fitresults_tb[best_iter]
        print target_batch, nllh[-1], fitresults_tb[best_iter]['success']

    bf_t = np.argmin(nllh)
    fr = fitresults[bf_t]
    bf_par = {tofit[i]: fr['x'][i] for i in range(len(tofit))}
    bf_par['target_batch'] = bf_t

    print 'best fitting target_batch size:', bf_t
    print 'bic:', bic(fr, pars)
    print fr

    return {'bf_par': bf_par,
            'nllh': fr['fun'],
            'bic': bic(fr, pars),
            'success': fr['success']}


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


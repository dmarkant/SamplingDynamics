from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, gamma
from cogmod.cpt import util
from mypy.explib.hau2008 import hau2008
from fitting import *

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
    eval_pow  = pars.get('eval_pow', 1.)

    th_shape  = pars.get('th_shape', 2.)
    th_scale  = pars.get('th_scale', 1.)

    t_batch = np.round(pars.get('target_batch', 5)) # target total sample size
    s_batch = pars.get('s_batch', 1.)

    p_guess   = pars.get('p_guess', 0.)

    # first evalute the trajectory
    samples = [0.]
    for trial, obs in enumerate(data['sampledata']):

        # evaluate the outcome
        samples.append(valuation([obs, data['outcomes'][trial]], eval_crit, eval_pow))
    pref = np.cumsum(samples)

    # on each trial, the probability of crossing the boundary is 
    # determined by the distribution over separation sizes
    p_stop = gamma.cdf(np.abs(pref), th_shape, scale=th_scale)
    p_stop[0] = 0.
    
    # probability of switching is based on streak count
    count_streak = count_streaks(data['sampledata'])
    p_stay = 1. / (1. + np.exp((np.array(count_streak) + 1 - t_batch) * s_batch))


    p_samp = 1 - p_stop
    d = np.array(data['sampledata'])
    p_sample_A = p_samp[1:] * (p_stay[1:] * (d==0) + (1 - p_stay[1:]) * (d==1))
    p_sample_A = np.concatenate(([.5], p_sample_A))

    p_sample_B = p_samp[1:] * (p_stay[1:] * (d==1) + (1 - p_stay[1:]) * (d==0))
    p_sample_B = np.concatenate(([.5], p_sample_B))

    # at end of sampling, give choice probabilities

    return {'pref': pref,
            'p_stop': p_stop,
            'p_sample_A': p_sample_A,
            'p_sample_B': p_sample_B}


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

    obj = pars['obj']

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

        p_stop = result['p_stop'][-1]
        llh_sampling += np.log(pfix(p_stop))

        #if _pars['data']['choice']==0:
        #    top = result['p_stop_choose_A'][-1]
        #else:
        #    top = result['p_stop_choose_B'][-1]
        #bottom = result['p_stop_choose_A'][-1] + result['p_stop_choose_B'][-1]
        #p_choice = top / bottom
        #llh_choice += np.log(pfix(p_choice))

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

    #pars = {'data': data,
    #        'obj': obj,
    #        'fitting': fitting}
    #for parname in fixed:
    #    pars[parname] = fixed[parname]

    def bic(f, pars):
        return 2 * f['fun'] + len(pars['fitting']) * np.log(np.sum([d['sampledata'].size + 1 for d in pars['data']]))


    counts = []
    for d in data:
        counts.append(len(d['sampledata']))
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
        init = [randstart(par) for par in pars['fitting']]
        f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')
        nllh.append(f['fun'])
        fitresults[target_batch] = f
        print target_batch, f['fun'], f['success']
    
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


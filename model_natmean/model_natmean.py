import numpy as np
from cogmod.cpt import util
from mypy.explib.hau2008 import hau2008
from fitting import *




def evaluation(obs, eval_crit, eval_pow):
    uA = util(obs[1], eval_pow) if obs[0]==0 else 0.
    uB = util(obs[1], eval_pow) if obs[0]==1 else 0.
    return uA - eval_crit - uB


def run(pars):

    verbose = pars.get('verbose', False)
    options = pars.get('options')
    data    = pars.get('data')

    rho       = pars.get('rho', .5)
    eval_crit = pars.get('eval_crit', 0)
    eval_pow  = pars.get('eval_pow', 1.)
    z         = pars.get('z', 0.)
    theta     = pars.get('theta')

    # add distribution over starting positions

    samples = [z]
    for trial, obs in enumerate(data['sampledata']):

        # evaluate the outcome
        samples.append(evaluation(obs, eval_crit, eval_pow))

    pref = np.cumsum(samples)[1:]

    p_resp = [1., 0.] if pref[-1] > 0 else [0., 1.]

    p_switch_t = rho * np.ones(len(data['sampledata']), float)
    p_switch_t[0] = 0. # the first observation will never be a switch

    # vector indicating whether preference state
    # has crossed the threshold defined by theta
    p_stop_t = np.array([1. * (pref > theta),
                         1. * (pref < -theta)])

    return {'p_resp': p_resp,
            'p_switch_t': p_switch_t,
            'p_stop_t': p_stop_t}



def loglik(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    data = pars['data']

    result = run(pars)

    sampledata = data['sampledata']
    llh = 0.
    for trial, obs in enumerate(sampledata):

        if trial==0 or obs[0]==sampledata[trial-1][0]:
            switched = 0.
        else:
            switched = 1.

        if (trial+1)==len(sampledata):
            stopped = 1.
        else:
            stopped = 0.

        # switched?
        if switched:
            llh += np.log(pfix(result['p_switch_t'][trial]))
        else:
            llh += np.log(pfix(1 - result['p_switch_t'][trial]))

        # continue or stop?
        if stopped:
            llh += np.log(pfix(np.sum(result['p_stop_t'], 0)[trial]))
        else:
            llh += np.log(pfix(1 - np.sum(result['p_stop_t'], 0)[trial]))

        # if stopped, which choice?
        if stopped:
            llh += np.log(pfix(result['p_resp'][data['choice']]))

    return -llh



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

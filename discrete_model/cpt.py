import numpy as np
from fitting import *
from copy import deepcopy


def util(outcome, pow_gain, pow_loss, w_loss):
    if outcome >= 0.:
        return (outcome ** pow_gain)
    else:
        return (-w_loss * ((-outcome) ** pow_loss))



def value(option, pow_gain, pow_loss, w_loss, w_p):

    v = 0.
    for outcome, prob in option:
        p_weighted = pweight(prob, w_p)

        if outcome >= 0:
            v += p_weighted * (outcome**pow_gain)
        else:
            v += p_weighted * (-w_loss * ((-outcome)**pow_loss))
    return v


def pweight(p, w):
    return pfix((p**w) / ((p**w + (1-p)**w) ** (1/w)))


def choice_prob(options, cpt_pars):
    vL = value(options[0], cpt_pars['pow_gain'], cpt_pars['pow_loss'], 1., cpt_pars['w_p'])
    vH = value(options[1], cpt_pars['pow_gain'], cpt_pars['pow_loss'], 1., cpt_pars['w_p'])
    s = cpt_pars['s']
    cp = np.exp(vH * s) / (np.exp(vH * s) + np.exp(vL * s))
    return cp


def loglik(value, pars):

    print 'llh'


def loglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    llh = []
    for gambledata in pars['data']:
        gpars = deepcopy

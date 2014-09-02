import numpy as np


def pfix(p):
    return np.min([np.max([p, 1e-5]), 1-(1e-5)])


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

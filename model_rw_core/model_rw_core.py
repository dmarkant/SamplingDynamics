import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power
from copy import deepcopy
from fitting import *
from cogmod.cpt import util_v, pweight_prelec
from time import time



def transition_probs(drift_fnc, state, pars):
    """Get transition probabilities given parameters and current
    state.

    drift_fnc -- pointer to function that will compute drift
    state -- index of current state
    tau -- time step size
    alpha -- relative prob of staying in current state (see Diederich
             and Busemeyer, 2003, p. 311)
    """
    alpha = pars.get('alpha', 1.3)
    tau   = pars.get('tau', 1.)

    dr = drift_fnc(state, pars)

    # drift must be bounded by -1, 1 to ensure probabilities
    if dr <= -1:
        dr = -.99999
    elif dr >= 1:
        dr = .99999

    p_down = (1./(2 * alpha)) * (1 - (dr) * np.sqrt(tau))
    p_up = (1./(2 * alpha)) * (1 + (dr) * np.sqrt(tau))
    p_stay = 1 - (1./alpha)

    try:
        assert np.round(np.sum([p_down, p_stay, p_up]), 5)==1.
    except AssertionError:
        print 'transition probabilities don\'t sum to 1'
        print [p_down, p_stay, p_up]
        print np.sum([p_down, p_stay, p_up])
    return [p_down, p_stay, p_up]


def transition_matrix_PQR(V, dv, drift_fnc, pars):
    """
    Transition matrix in arranged in PQR form

    V -- discrete state space
    dv -- step size
    """
    gamma = pars.get('gamma', 0.)

    m = len(V)
    tm_pqr = np.zeros((m, m), float)
    tm_pqr[0,0] = 1.
    tm_pqr[1,1] = 1.
    vi_pqr = []

    start = np.array([[0, m - 1], range(1, m - 1)])
    for outer in start:
        for inner in outer:
            vi_pqr.append(inner)
    vi_pqr = np.array(vi_pqr)
    V_pqr = V[vi_pqr] # sort state space

    # if there is state-dependent weighting, compute
    # transition probabilities for each state. Otherwise,
    # use same transition probabilities for everything
    if gamma == 0.:
        tp = np.tile(transition_probs(drift_fnc, 1, pars), (m - 2, 1))
    else:
        tp = np.array([transition_probs(drift_fnc, i*dv, pars) for i in range(1, m - 1)])

    # construct PQR row by row
    for i in range(1, m - 1):
        row = np.where(V_pqr==V[i])[0][0]
        ind_pqr = np.array([np.where(V_pqr==V[i-1])[0][0], np.where(V_pqr==V[i])[0][0], np.where(V_pqr==V[i+1])[0][0]])
        tm_pqr[row, ind_pqr] = tp[i-1]

    return tm_pqr




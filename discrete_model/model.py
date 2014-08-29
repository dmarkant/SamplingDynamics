import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power
from copy import deepcopy


def pfix(p):
    return np.min([np.max([p, 1e-5]), 1-(1e-5)])


def transition_probs(v, delta, sigma, tau, gamma):
    """Get transition probabilities given parameters and current
    state.

    v -- current state
    delta -- drift parameter
    gamma -- state-dependent weight
    sigma -- diffusion parameter
    tau -- time step size
    """
    # state-dependent weight depends on current state and gamma
    sdw = gamma * v
    p_down = .5 * (1 - ((delta - sdw)/sigma) * np.sqrt(tau))
    p_up = .5 * (1 + ((delta - sdw)/sigma) * np.sqrt(tau))
    p_stay = 1 - p_down - p_up
    return [p_down, p_stay, p_up]


def transition_matrix_PQR(V, dv, delta, sigma, tau, gamma):
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

    # construct PQR row by row
    for i in range(1, m - 1):
        row = np.where(V_pqr==V[i])[0][0]
        ind_pqr = np.array([np.where(V_pqr==V[i-1])[0][0], np.where(V_pqr==V[i])[0][0], np.where(V_pqr==V[i+1])[0][0]])
        tm_pqr[row, ind_pqr] = transition_probs(i*dv, delta, sigma, tau, gamma)

    return tm_pqr


def loglik(value, args):

    verbose = args.get('verbose', False)

    pars = deepcopy(args)
    fitting = pars['fitting']
    if verbose: print 'evaluating:'
    for i, k in enumerate(fitting):
        pars[k] = value[i]
        if verbose: print '  %s=%s' % (k, pars[k])

    if pars['theta'] <= 0:
        return np.inf
    else:
        result = run(pars)

        data = pars['data']
        pstop = result['p_stop_t']

        llh = 0.
        for obs in data:
            choice, t = obs
            llh += -np.log(pfix(pstop[t, choice]))

        if verbose: print '  llh: %s' % llh
        return llh


def run(pars):

    theta = pars.get('theta', 4)     # boundaries
    #m     = pars.get('m', 21)        # number of states
    dv    = pars.get('dv', .1)       # state space step size
    delta = pars.get('delta', .1)    # drift parameter
    sigma = pars.get('sigma', 1.)    # diffusion parameter
    gamma = pars.get('gamma', 0.)    # state-dependent weight on drift
    max_T = pars.get('max_T', 100)   # range of timesteps to evaluate over
    dt    = pars.get('dt', 1.)       # size of timesteps to evaluate over
    z_w   = pars.get('z_width', .1)

    #dv = (2*theta)/float(m-1)
    tau = (dv**2)/(sigma**2)

    # state space
    V = np.round(np.arange(-theta, theta+(dv/2.), dv), 4)
    vi = range(len(V))
    m = len(V)

    if 'Z' in pars:
        Z = np.matrix(pars.get('Z'))
    else:
        # uniform interval for starting position
        zhw = np.floor((m - 2) * z_w)
        Z = np.zeros(m - 2)
        Z[(len(Z)/2-zhw):(len(Z)/2+zhw+1)] = 1.
        Z = np.matrix(Z/float(sum(Z)))

    # transition matrix
    tm_pqr = transition_matrix_PQR(V, dv, delta, sigma, tau, gamma)
    Q = tm_pqr[2:,2:]
    I = np.eye(m - 2)
    R = np.matrix(tm_pqr[2:,:2])
    IQ = np.matrix(linalg.inv(I - Q))


    # time steps for evaluation
    T = np.arange(0., max_T, dt)
    N = map(int, np.floor(T/tau) + 1)


    # predicted response probabilities
    # 1. overall response prob
    resp_prob = Z * (IQ * R)

    # 2. response probability over time
    resp_prob_t = np.array([Z * (matrix_power(Q, n - 1) * R) for n in N]).reshape((len(N), 2))

    # 3. cumulative response probability over time
    #resp_prob_cump = resp_prob_t.cumsum(axis=0)

    # predicted stopping points, conditional on choice
    # 1. mean stopping point
    p_tsteps = (Z * (IQ * IQ) * R) / resp_prob

    # 2. probability of stopping over time
    p_stop_cond = np.array([(Z * ((matrix_power(Q, n - 1) * R)))/resp_prob for n in N]).reshape((len(N), 2))

    # 3. cumulative probability of stopping over time
    p_stop_cond_cump = np.array([Z * IQ * (I - matrix_power(Q, n)) * R / resp_prob for n in N]).reshape((len(N), 2))

    return {'T': T,
            'resp_prob': np.array(resp_prob)[0],
            #'resp_prob_t': resp_prob_t,
            #'resp_prob_cump': resp_prob_cump,
            'p_tsteps': p_tsteps,
            'p_stop_t': p_stop_cond,
            'p_stop_t_cump': p_stop_cond_cump}


if __name__ == '__main__':

    # run test
    pars = {'theta': 4,
            'dv': .5,
            'delta': .05,
            'sigma': 1.,
            'gamma': 0.}

    result = run(pars)
    print result


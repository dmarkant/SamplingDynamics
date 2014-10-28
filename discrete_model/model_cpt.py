import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power
from copy import deepcopy
from fitting import *
from cogmod.cpt import util, pweight_prelec


def option_stats(options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma):

    probs = []
    utils = []
    eu    = []
    evar =  []

    for opt_i, option in enumerate(options):

        probs.append([pweight_prelec(prob, prelec_elevation, prelec_gamma) for (outcome, prob) in option])
        utils.append([util(outcome, pow_gain, pow_loss, w_loss) for (outcome, prob) in option])

        eu.append(np.array([probs[opt_i][i] * utils[opt_i][i] for i in range(len(probs[opt_i]))]))

        evar.append(np.sum([probs[opt_i][i] * (utils[opt_i][i] ** 2) for i in range(len(probs[opt_i]))]) - np.sum(eu[opt_i]) ** 2)

    outs =  np.outer(utils[0] - eu[0], utils[1] - eu[1])
    ps   = np.outer(probs[0], probs[1])
    cov = np.sum(np.multiply(outs, ps))
    pooledvar = np.sum(evar) - 2 * cov
    seu = np.sum(eu, axis=1)

    #print '--------'
    #print options[0]
    #print options[1]


    #print 'utils:', utils
    #print 'eu:', eu
    #print 'seu:', seu
    #print 'evar:', evar
    #print 'cov:', cov
    #print 'poolvar:', pooledvar
    #print 'sigma:', np.sqrt(pooledvar)

    try:
        assert np.isnan(cov)==False and pooledvar > 0
    except:
        pass
        print 'problem with variance in drift rate!'

    return seu, pooledvar


def drift(options, v, delta, gamma, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma):
    """
    v -- current state
    """
    sdw = gamma * v

    seu, pooledvar = option_stats(options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma)

    return delta * (seu[1] - seu[0]) / (np.sqrt(pooledvar))



def transition_probs(v, delta, tau, gamma, alpha, options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma):
    """Get transition probabilities given parameters and current
    state.

    v -- current state
    delta -- drift parameter
    gamma -- state-dependent weight
    sigma -- diffusion parameter
    tau -- time step size
    alpha -- relative prob of staying in current state (see Diederich
             and Busemeyer, 2003, p. 311)
    """
    dr = drift(options, v, delta, gamma, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma)

    if dr <= -1:
        dr = -.9999
    elif dr >= 1:
        dr = .9999

    p_down = (1./(2 * alpha)) * (1 - (dr) * np.sqrt(tau))
    p_up = (1./(2 * alpha)) * (1 + (dr) * np.sqrt(tau))
    p_stay = 1 - (1./alpha)

    try:
        assert np.round(np.sum([p_down, p_stay, p_up]), 5)==1.
    except AssertionError:
        print 'GAH'
        print np.sum([p_down, p_stay, p_up])
    return [p_down, p_stay, p_up]


def transition_matrix_PQR(V, dv, delta, tau, gamma, alpha, options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma):
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
        tm_pqr[row, ind_pqr] = transition_probs(i*dv, delta, tau, gamma, alpha, options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma)

    return tm_pqr


def run(pars):

    verbose = pars.get('verbose', False)

    options = pars.get('options')
    pow_gain = pars.get('pow_gain', 1.)
    pow_loss = pars.get('pow_loss', pow_gain) # if not provided, assume
                                              # same as pow_gain
    w_loss = pars.get('w_loss', 1.)
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)

    p_switch = pars.get('p_switch', 0.5)

    delta = pars.get('delta', 1.)    # drift scaling factor

    theta = np.float(np.round(pars.get('theta', 5)))     # boundaries

    sigma = pars.get('sigma', 1.)    # diffusion parameter
    gamma = pars.get('gamma', 0.)    # state-dependent weight on drift
    max_T = pars.get('max_T', 100)   # range of timesteps to evaluate over
    dt    = pars.get('dt', 1.)       # size of timesteps to evaluate over
    alpha = pars.get('alpha', 1.3)   # for transition probs, controls the
                                     # stay probability (must be > 1)

    dv    = pars.get('dv', 1.)
    tau   = (dv**2)/(sigma**2)       # with default settings, equal to 1.

    # create state space
    V = np.round(np.arange(-theta, theta+(dv/2.), dv), 4)
    vi = range(len(V))
    m = len(V)


    if 'Z' in pars:
        # starting vector was provided
        Z = np.matrix(pars.get('Z'))

    elif 'z_width' in pars:
        # use interval of uniform probability for starting position,
        # with width set by z_width
        z_w   = pars.get('z_width')
        z_w = z_w / 2.
        zhw = np.floor((m - 2) * z_w)
        Z = np.zeros(m - 2)
        Z[(len(Z)/2-zhw):(len(Z)/2+zhw+1)] = 1.
        if verbose:
            pass
            #print '  actual halfwidth of starting interval:', np.round(len(np.where(Z==1.)[0]) / float(len(Z)),2)

        Z = np.matrix(Z/float(sum(Z)))
    elif 'z_temp' in pars:
        # use softmax-transformed distance from unbiased point
        z_temp   = pars.get('z_temp')
        Z = np.exp(-np.abs(V[1:-1]) * (1/float(z_temp)))
        Z = np.matrix(Z / np.sum(Z))
        if verbose:
            pass
            #print 'z_temp:', z_temp
    else:
        # otherwise start with unbiased starting position
        Z = np.zeros(m - 2)
        Z[len(Z)/2] = 1.
        Z = np.matrix(Z)

    # transition matrix
    tm_pqr = transition_matrix_PQR(V, dv, delta, tau, gamma, alpha, options, pow_gain, pow_loss, w_loss, prelec_elevation, prelec_gamma)

    Q = tm_pqr[2:,2:]
    I = np.eye(m - 2)
    R = np.matrix(tm_pqr[2:,:2])
    IQ = np.matrix(linalg.inv(I - Q))

    # time steps for evaluation
    T = np.arange(1., max_T + 1, dt)
    N = map(int, np.floor(T/tau))

    states_t = np.array([Z * (matrix_power(Q, n - 1)) for n in N]).reshape((len(N), m - 2))

    # 1. overall response probabilities
    resp_prob = Z * (IQ * R)

    # 2. response probability over time
    resp_prob_t = np.array([Z * (matrix_power(Q, n - 1) * R) for n in N]).reshape((len(N), 2))

    # 3. cumulative response probability over time
    #resp_prob_cump = resp_prob_t.cumsum(axis=0)

    # 1. predicted stopping points, conditional on choice
    p_tsteps = (Z * (IQ * IQ) * R) / resp_prob

    # 2. probability of stopping over time
    p_stop_cond = np.array([(Z * ((matrix_power(Q, n - 1) * R)))/resp_prob for n in N]).reshape((len(N), 2))

    # 3. cumulative probability of stopping over time
    p_stop_cond_cump = np.array([Z * IQ * (I - matrix_power(Q, n)) * R / resp_prob for n in N]).reshape((len(N), 2))

    return {'T': T,
            'states_t': states_t,
            'resp_prob': np.array(resp_prob)[0],
            'resp_prob_t': resp_prob_t,
            'p_tsteps': p_tsteps,
            'p_stop_t': p_stop_cond,
            'p_stop_t_cump': p_stop_cond_cump}


def loglik(value, args):

    verbose = args.get('verbose', False)

    pars = deepcopy(args)
    fitting = pars['fitting']
    if verbose: print 'evaluating:'
    for i, k in enumerate(fitting):
        pars[k] = value[i]
        if verbose: print '  %s=%s' % (k, pars[k])


    if 'z_temp' in pars and pars['z_temp'] <= 0.:
        return np.inf
    elif 'z_width' in pars and (pars['z_width']<.05 or pars['z_width']>1):
        return np.inf
    elif pars['theta'] < 1. or pars['delta']<-1. or pars['delta']>1.:
        return np.inf
    else:
        gpars = deepcopy(pars)
        gpars.update({'data': pars['data']['samplesize'],
                      'max_T': pars['data']['max_t'],
                      'options': pars['data']['options']})

        result = run(gpars)

        data = pars['data']['samplesize']
        pchoice = result['resp_prob_t']
        pstop = result['p_stop_t']

        llh = 0.
        for obs in data:
            choice, t = obs
            llh += -1 * (np.log(pfix(pstop[t, choice])) + np.log(pfix(result['resp_prob'][choice])))
            #llh += -1 * (np.log(pfix(pstop[t, choice])))

        if verbose: print '  llh: %s' % llh
        return llh


def loglik_across_gambles(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    llh = []
    for gambledata in pars['data']:

        gpars = deepcopy(pars)
        gpars.update({'data': gambledata['samplesize'],
                      'max_T': gambledata['max_t'],
                      'options': gambledata['options']})

        result = run(gpars)

        g_llh = 0.
        for obs in gpars['data']:
            choice, t, grp = obs
            g_llh += -1 * (np.log(pfix(result['p_stop_t'][t, choice])) + np.log(pfix(result['resp_prob'][choice])))
        llh.append(g_llh)

    if verbose: print '  llh: %s' % np.sum(llh)
    return np.sum(llh)


def loglik_across_gambles_by_group(value, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(pars): return np.inf

    llh = []
    for gambledata in pars['data']:
        gpars_0 = deepcopy(pars)
        gpars_1 = deepcopy(pars)
        gpars_0.update({'data': gambledata['samplesize'],
                        'max_T': gambledata['max_t'],
                        'options': gambledata['options']})
        gpars_1.update({'data': gambledata['samplesize'],
                        'max_T': gambledata['max_t'],
                        'options': gambledata['options']})

        if 'theta(0)' in pars:
            gpars_0.update({'theta': gpars_0['theta(0)']})
            gpars_1.update({'theta': gpars_1['theta(1)']})

        if 'pow_gain(0)' in pars:
            gpars_0.update({'pow_gain': gpars_0['pow_gain(0)']})
            gpars_1.update({'pow_gain': gpars_1['pow_gain(1)']})

        if 'pow_loss(0)' in pars:
            gpars_0.update({'pow_loss': gpars_0['pow_loss(0)']})
            gpars_1.update({'pow_loss': gpars_1['pow_loss(1)']})

        if 'delta(0)' in pars:
            gpars_0.update({'delta': gpars_0['delta(0)']})
            gpars_1.update({'delta': gpars_1['delta(1)']})

        if 'z_temp(0)' in pars:
            gpars_0.update({'z_temp': gpars_0['z_temp(0)']})
            gpars_1.update({'z_temp': gpars_1['z_temp(1)']})

        if 'prelec_elevation(0)' in pars:
            gpars_0.update({'prelec_elevation': gpars_0['prelec_elevation(0)']})
            gpars_1.update({'prelec_elevation': gpars_1['prelec_elevation(1)']})


        result = [run(gpars_0), run(gpars_1)]

        g_llh = 0.
        for obs in gpars_0['data']:
            choice, t, grp = obs
            g_llh += -1 * (np.log(pfix(result[grp]['p_stop_t'][t, choice])) + np.log(pfix(result[grp]['resp_prob'][choice])))
        llh.append(g_llh)

    #print llh
    if verbose: print '  llh: %s' % g_llh
    return np.sum(llh)




if __name__ == '__main__':


    options = [[[  4.        ,   0.5       ],
                [ 12.        ,   0.26315789],
                [  0.        ,   0.23684211]],
               [[  1.  ,   0.41],
                [  0.  ,   0.28],
                [ 18.  ,   0.31]]]

    # run test
    pars = {'theta': 4,
            'options': options,
            'delta': .05}

    result = run(pars)
    print result


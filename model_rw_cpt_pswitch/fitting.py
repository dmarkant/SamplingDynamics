import numpy as np
from copy import deepcopy


BOUNDS = {'theta': [1, np.inf],
          'delta': [0, np.inf],
          'z_temp': [0, np.inf],
          'prelec_elevation': [0, np.inf],
          'prelec_gamma': [0, np.inf],
          'pow_gain': [0, np.inf],
          'pow_loss': [0, np.inf],
          'p_switch': [0, 1],
          's': [0, np.inf]} # temp for cpt softmax


def get_bounds(parname):
    if '(' in parname:
        return BOUNDS[parname[:parname.index('(')]]
    else:
        return BOUNDS[parname]


def pfix(p):
    return np.min([np.max([p, 1e-5]), 1-(1e-5)])


def randstart(parname):
    if parname is 'theta':
        return np.random.randint(2, 8)
    elif parname is 'delta':
        return 1.
    elif parname is 'prelec_elevation':
        return np.random.random()
    elif parname is 'prelec_gamma':
        return np.random.random()
    elif parname is 'p_switch':
        return 0.5
    else:
        return np.random.random()


def unpack(value, args):
    verbose = args.get('verbose', False)
    pars = deepcopy(args)
    fitting = pars['fitting']
    if verbose: print 'evaluating:'
    for i, k in enumerate(fitting):
        pars[k] = value[i]
        if verbose: print '  %s=%s' % (k, pars[k])

    return pars, fitting, verbose


def outside_bounds(pars):
    outside = False
    if 'bounds' in pars:
        bounds = pars['bounds']
    else:

        bounds = [get_bounds(p) for p in pars['fitting']]
    for i, k in enumerate(pars['fitting']):
        if pars[k] < bounds[i][0] or pars[k] > bounds[i][1]:
            outside = True
    return outside


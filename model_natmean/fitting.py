import numpy as np
from copy import deepcopy


BOUNDS = {'eval_crit': [-np.inf, np.inf],
          'eval_pow': [0, np.inf],
          'z_mu': [-np.inf, np.inf],
          'z_sd': [0, np.inf],
          'theta': [-np.inf, np.inf],
          'th_scale': [0, np.inf],
          'th_shape': [0, np.inf],
          'target_batch': [0, np.inf],
          's_batch': [0, 10],
          'rho': [0., 1.],
          'alpha': [1, 1000],
          'beta': [0, 1000],
          'mu': [0, np.inf],
          'sd': [0, np.inf],
          'p_guess': [0, 1]}


def get_bounds(parname):
    if '(' in parname:
        return BOUNDS[parname[:parname.index('(')]]
    else:
        return BOUNDS[parname]


def pfix(p):
    return np.min([np.max([p, 1e-5]), 1-(1e-5)])


def randstart(parname):
    if parname is 'theta':
        return np.random.randint(0, 20)
    elif parname is 'th_scale':
        return np.random.random() * 2
    elif parname is 'th_shape':
        return np.random.random() * 10
    elif parname is 'eval_crit':
        return np.random.randint(-20, 20)
    elif parname is 'eval_pow':
        return np.random.random()
    elif parname is 'z_mu':
        return 0.
    elif parname is 'z_sd':
        return np.random.random() * 10.
    elif parname is 'rho':
        return np.random.random()
    elif parname is 's_batch':
        return np.random.random()
    elif parname is 'alpha':
        return 1+np.random.random()*10
    elif parname is 'beta':
        return np.random.random()*10
    elif parname is 'mu':
        return np.random.randint(1, 20)
    elif parname is 'sd':
        return np.random.randint(1, 10)
    elif parname is 'p_guess':
        return np.random.random()
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


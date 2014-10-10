import numpy as np
from copy import deepcopy


BOUNDS = {'target': [0, np.inf],
          's': [0, 10.]}


def pfix(p):
    return np.min([np.max([p, 1e-5]), 1-(1e-5)])


def get_bounds(parname):
    if '(' in parname:
        return BOUNDS[parname[:parname.index('(')]]
    else:
        return BOUNDS[parname]


def randstart(parname):
    if parname is 'target':
        return np.random.randint(1, 20)
    elif parname is 's':
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


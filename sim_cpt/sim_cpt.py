import sys
sys.path.append('../')

import pickle, os
from copy import deepcopy
import numpy as np
import pandas as pd
from helpers import *
from scipy.stats import norm, gamma, lognorm, geom
from scipy.optimize import minimize

from mypy.explib.frey2014 import frey2014
from model_rw_cpt import model_cpt
from model_rw_cpt.fitting import *
from cogmod.fitting import bic

from mypy.sim import sim_id_str, pickle_sim_result, unpickle_sim_result


GAMBLE_DATA_PATH = '../dfe_by_gamble.csv'
SUBJECT_DATA_PATH = '../dfe_by_game.csv'
OUTPUT_PATH = './output'


def get_options(gid):
    gdata = data_by_gamble[data_by_gamble['gid']==gid]
    L = np.array([gdata[['L_x1', 'L_p1']].values, gdata[['L_x2', 'L_p2']].values, gdata[['L_x3', 'L_p3']].values]).reshape((3,2))
    H = np.array([gdata[['H_x1', 'H_p1']].values, gdata[['H_x2', 'H_p2']].values, gdata[['H_x3', 'H_p3']].values]).reshape((3,2))
    return [L, H]




def init():
    global data, data_by_gamble, gamble_lab_srt, switchfreq, SUBJ
    data_by_gamble = pd.read_csv(GAMBLE_DATA_PATH)

    data = pd.read_csv(SUBJECT_DATA_PATH)
    gamble_lab = data['gamble_lab'].unique()

    # sort by problem type
    gambles = data[data['partid']==1]
    gambles_srt = gambles.sort(['domain', 'pairtype', 'session'])
    gamble_lab_srt = gambles_srt['gamble_lab']

    SUBJ = data['partid'].unique()


def subject_fitdata(sid):

    fitdata = []
    for gid in gamble_lab_srt:
        gdata = data[(data['partid']==sid) & (data['gamble_lab']==gid)]
        tofit = []
        for obs in gdata[['decision', 'samplesize', 'group']].values:
            choice = 0 if obs[0]=='L' else 1
            group = 0 if obs[2]=='old' else 1
            samplesize = obs[1]
            tofit.append([choice, samplesize, group])
        tofit = np.array(tofit)

        if len(tofit) > 0:

            max_t = tofit[0][1] + 1

            fitdata.append({'gid': gid,
                            'options': get_options(gid),
                            'max_t': max_t,
                            'data': tofit})
    return fitdata


def fit_subject(sid, thetas=None, fitting=None, init=None):
    print 'Fitting subject %s' % sid

    fitdata = subject_fitdata(sid)
    if thetas is None:
        thetas = range(2, 12)
    if fitting is None:
        fitting = ['delta', 'z_temp', 'pow_gain']
    fth = {}
    for theta in thetas:
        pars = {'data': fitdata,
                'theta': theta,
                'fitting': fitting}
        if init is None:
            init = [1., 1., 1.]
        fth[theta] = minimize(model_cpt.loglik_across_gambles, init, (pars,), method='Nelder-Mead')
        print 'theta:', theta
        print fth[theta]

    bf_theta = thetas[np.argmin([fth[th]['fun'] for th in thetas])]
    f = fth[bf_theta]
    return {'bf_theta': bf_theta, 'f': f, 'fitting': fitting}


def predict(sid, fitresult):
    """
    Given a set of parameters, predict sample size distribution
    and response proportion.
    """
    rp = {}
    results = {}
    fitdata = subject_fitdata(sid)
    for fd in fitdata:
        p = {'theta': fitresult['bf_theta'],
             'options': fd['options']}
        for i, parname in enumerate(fitresult['fitting']):
            p[parname] = fitresult['f']['x'][i]

        results[fd['gid']] = model_cpt.run(p)
    return results


def show_result(sid, fitresult):

    results = predict_emp(sid, fitresult)

    b = bic(fitresult['f']['fun'],
        len(fitresult['fitting']) + 1,
        len(subject_fitdata(sid)) * 2)

    print 'subject %s' % sid
    print 'bic: %s' % b
    print 'bf_par:'
    print '   theta: %s' % fitresult['bf_theta']
    for i, pname in enumerate(fitresult['fitting']):
        print '   %s: %s' % (pname, fitresult['f']['x'][i])

    # predicted choice proportions
    dec = [[1 if dec=='H' else 0 for dec in data[(data['partid']==sid) & (data['gamble_lab']==gid)]['decision'].values] for gid in gamble_lab_srt]
    dec = np.array(dec).transpose()[0]
    rp = np.array([results[gid]['resp_prob'][1] for gid in gamble_lab_srt]).transpose()
    rp_by_type = [np.mean(rp[:21]), np.mean(rp[21:42]), np.mean(rp[42:63]), np.mean(rp[63:])]
    dec_by_type = [np.mean(dec[:21]), np.mean(dec[21:42]), np.mean(dec[42:63]), np.mean(dec[63:])]

    # predicted sample sizes
    predicted_L = np.array([results[gid]['p_tsteps'][0] for gid in gamble_lab_srt])
    predicted_H = np.array([results[gid]['p_tsteps'][1] for gid in gamble_lab_srt])
    arr = data[(data['partid']==sid)][['decision', 'samplesize']].values
    chose_L = arr[arr[:,0]=='L'][:,1]
    chose_H = arr[arr[:,0]=='H'][:,1]
    max_ss = max([max(chose_L), max(chose_H)]) + 5

    fig, ax = plt.subplots(1, 3, figsize=(20,5))

    ax[0].plot(rp_by_type, '-x', color='gray', label='predicted p(A)')
    ax[0].plot(dec_by_type, '-o', color='black', label='observed prop(A)')
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(-1, 4)
    ax[0].legend()

    for gid in gamble_lab_srt:
        ax[1].plot(results[gid]['p_stop_t'][0], color='black', alpha=.1)
        ax[2].plot(results[gid]['p_stop_t'][1], color='black', alpha=.1)

    ax[1].hist(chose_L, bins=10, alpha=0.5, weights=np.zeros_like(chose_L) + 1. / chose_L.size, color='gray')
    #ax[1].hist(predicted_L, bins=10, alpha=0.5, weights=np.zeros_like(predicted_L) + 1. / predicted_L.size, color='blue')
    ax[1].set_xlim(0, max_ss)

    ax[2].hist(chose_H, bins=10, alpha=0.5, weights=np.zeros_like(chose_H) + 1. / chose_H.size, color='gray')
    #ax[2].hist(predicted_H, bins=10, alpha=0.5, weights=np.zeros_like(predicted_H) + 1. / predicted_H.size, color='blue')
    ax[2].set_xlim(0, max_ss)

    plt.show()


def save_result(sid, fitresult):

    fitting = deepcopy(fitresult['fitting'])
    if 'theta' not in fitting:
        fitting.append('theta')

    fitting = {p: None for p in fitting}

    pickle_sim_result(name='sim_cpt',
                      result_id=sid,
                      par=fitting,
                      result=fitresult)


def load_result(sid, fitting=[], fixed=[]):

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # add theta to parameter list
    if 'theta' not in fitting:
        fitting.append('theta')
    fitting = {p: None for p in fitting}

    r = unpickle_sim_result(name='sim_cpt',
                            result_id=sid,
                            par=fitting,
                            outdir=dirname)

    return r


if __name__ == '__main__':

    init()
    sids = SUBJ
    fitting = ['delta', 'z_temp', 'pow_gain']

    for sid in sids[5:]:

        result = fit_subject(sid,
                             thetas=range(1, 15),
                             fitting=fitting,
                             init=[1., 1., 1.])

        save_result(sid, result)


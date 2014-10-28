PARSETS = [['theta'],
           ['theta', 'prelec_elevation'],
           ['theta', 'prelec_elevation', 'pow_gain'],
           ['theta', 'prelec_elevation', 'pow_gain', 'pow_loss'],
           ['theta', 'prelec_gamma'],
           ['theta', 'prelec_gamma', 'pow_gain'],
           ['theta', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'prelec_elevation', 'prelec_gamma'],
           ['theta', 'prelec_elevation', 'prelec_gamma', 'pow_gain'],
           ['theta', 'prelec_elevation', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'pow_gain'],
           ['theta', 'pow_gain', 'pow_loss'],
           ['theta', 'delta'],
           ['theta', 'delta', 'prelec_elevation'],
           ['theta', 'delta', 'prelec_elevation', 'pow_gain'],
           ['theta', 'delta', 'prelec_elevation', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'prelec_gamma'],
           ['theta', 'delta', 'prelec_gamma', 'pow_gain'],
           ['theta', 'delta', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'prelec_elevation', 'prelec_gamma'],
           ['theta', 'delta', 'prelec_elevation', 'prelec_gamma', 'pow_gain'],
           ['theta', 'delta', 'prelec_elevation', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'pow_gain'],
           ['theta', 'delta', 'pow_gain', 'pow_loss'],
           ['theta', 'z_temp'],
           ['theta', 'z_temp', 'prelec_elevation'],
           ['theta', 'z_temp', 'prelec_elevation', 'pow_gain'],
           ['theta', 'z_temp', 'prelec_elevation', 'pow_gain', 'pow_loss'],
           ['theta', 'z_temp', 'prelec_gamma'],
           ['theta', 'z_temp', 'prelec_gamma', 'pow_gain'],
           ['theta', 'z_temp', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'z_temp', 'prelec_elevation', 'prelec_gamma'],
           ['theta', 'z_temp', 'prelec_elevation', 'prelec_gamma', 'pow_gain'],
           ['theta', 'z_temp', 'prelec_elevation', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'z_temp', 'pow_gain'],
           ['theta', 'z_temp', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'z_temp'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation', 'pow_gain'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'z_temp', 'prelec_gamma'],
           ['theta', 'delta', 'z_temp', 'prelec_gamma', 'pow_gain'],
           ['theta', 'delta', 'z_temp', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation', 'prelec_gamma'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation', 'prelec_gamma', 'pow_gain'],
           ['theta', 'delta', 'z_temp', 'prelec_elevation', 'prelec_gamma', 'pow_gain', 'pow_loss'],
           ['theta', 'delta', 'z_temp', 'pow_gain'],
           ['theta', 'delta', 'z_temp', 'pow_gain', 'pow_loss']
           ]

fitlog = 'fits_hau_study1_gridTheta.txt'

fitresults_grid_theta = {4: {}, 5: {}, 6: {}}



def fit():
    """Fit ideal observer model for one game for one subject"""

    # load necessary data
    try:
        s = Subject(subj)
        X = s.samples(game=game)
        board = s.gameboard(game=game)
    except:
        print "failed to read data for subj %s game %s" % (subj, game)
        return

    model = IdealObserver
    id = [["subj", subj],
          ["game", game]]

    init = { #"hspace": hspace,
             "subj": subj,
             "game": game,
             "board": board,
             "obs": X,
             "loadfromfile": True,
             "samplingnorm": samplingnorm,
           }

    fixed = { "samplingnorm": samplingnorm,
            }

    par = {"d": [0., 10.]}

    name = MODEL_ID
    s = Sim(logfile=logfile,
              rootdir=outputdir,
              name=name,
              model=model,
              id=id,
              init=init,
              fixed=fixed,
              par=par,
              nruns = 1,
              quiet = False
              )

    s()




if __name__ == '__main__':


    for theta in [5, 4, 6]:
        for freepar in PARSETS[1:]:

            freepar_notheta = [pname for pname in freepar if pname!='theta']

            pars = {'data': fitdata,
                    'fitting': freepar_notheta,
                    'theta': theta,
                    'bounds': [BOUNDS[p] for p in freepar_notheta]
            }
            init = map(randstart, freepar_notheta)
            f = minimize(loglik_across_gambles, init, (pars,), method='Nelder-Mead')
            b = bic(f, pars)

            fitresults_grid_theta[theta][tuple(freepar)] = [freepar_notheta, f, b]


            outstr = '\nFixed:\ttheta=%s\n' % theta
            outstr += 'Fitting:\t'
            for pname in freepar_notheta:
                outstr += '%s\t' % pname
            outstr += '\n'
            outstr += 'Success:\t%s\n' % f['success']
            outstr += 'NegLLH :\t%s\n' % f['fun']
            outstr += 'Value  :\t'
            for x in f['x']:
                outstr += '%s\t' % x
            outstr += '\n'
            outstr += 'BIC    :\t%s\n' % b
            print outstr
            with open(fitlog, 'a') as f:
                f.write(outstr)

            p_resp = []
            for i, fd in enumerate(fitdata):
                pred_pars = {'data': fd['samplesize'],
                            'max_T': fd['max_t'],
                            'theta': theta,
                            'options': fd['options']}
                for par_i, par_name in enumerate(freepar_notheta):
                    pred_pars[par_name] = fitresults_grid_theta[theta][tuple(freepar)][1]['x'][par_i]
                result = run(pred_pars)
                p_resp.append(result['resp_prob'][1])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1,7), [np.mean(choices(1, pid)) for pid in range(6)], '-o', label='observed')
            ax.plot(range(1,7), p_resp, '-x', label='best fit model')
            ax.set_xlim([0, 7])
            ax.set_ylim(0, 1)
            ax.legend()
            plt.show()

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from constants import N_C

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    if gamma == 1:
        return np.log(values)
    else:
        return values**(1-gamma) / (1-gamma)


def exp_val_new(y, savings_incr, grid_w, v, n_sim):

    COH = np.zeros((n_sim, N_C))
    COH[:] = np.squeeze(savings_incr)
    COH += y[None].T

    COH[COH > grid_w[-1]] = grid_w[-1]
    COH[COH < grid_w[0]] = grid_w[0]

    spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

    # p = mp.Pool(processes=mp.cpu_count())
    # v_w = p.apply(spline, args=(COH,))
    # p.close()

    v_w = np.zeros((n_sim, N_C))
    for i in range(n_sim):
        v_w[i, :] = spline(COH[i, :])

    ev = v_w.mean(axis=0)
    return ev[None].T


def export_incomes(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg, gamma):
    term = int(param_pair[0])
    rho = param_pair[1]


    # adj income
    #SIDHYA CHANGE
    adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim, alt_deg)

    pd.DataFrame(data=adj_income).to_csv('adj_income.csv')
from scipy.interpolate import CubicSpline
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
from constants import *
import multiprocessing as mp

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    if gamma == 1:
        return np.log(values)
    else:
        return values**(1-gamma) / (1-gamma)


def cal_income(coeffs):
    coeff_this_group = coeffs.loc[education_level[AltDeg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(START_AGE, RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    return income


def read_input_data(income_fp, mortal_fp):
    age_coeff_and_var = pd.ExcelFile(income_fp)
    # age coefficients
    age_coeff = pd.read_excel(age_coeff_and_var, sheet_name='Coefficients')

    # decomposed variance
    std = pd.read_excel(age_coeff_and_var, sheet_name='Variance', header=[1, 2])
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

    # conditional survival probabilities
    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)

    return age_coeff, std, cond_prob


def adj_income_process(income, sigma_perm, sigma_tran, INIT_DEBT, N_SIM):
    # generate random walk and normal r.v.
    np.random.seed(0)
    rn_perm = np.random.normal(MU, sigma_perm, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    np.random.seed(1)
    rn_tran = np.random.normal(MU, sigma_tran, (N_SIM, RETIRE_AGE - START_AGE + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income)

    # - retirement
    ret_income_vec = ret_frac[AltDeg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # unemployment risk
    # generate bernoulli random variable
    np.random.seed(2)
    p = 1 - unempl_rate[AltDeg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - START_AGE + 1, N_SIM)).astype(float)
    r[r == 0] = unemp_frac[AltDeg]
    ones = np.ones((END_AGE - RETIRE_AGE, N_SIM))
    bern = np.append(r, ones, axis=0)
    Y = np.multiply(inc_with_inc_risk, bern.T)

    # adjust income with debt repayment
    D = np.zeros(Y.shape)
    D[:, 0] = INIT_DEBT

    P = (Y - PL)
    P = np.where(P < 0, 0, P)
    P = P * 0.15   # income share percentage !!!
    P[:, 20:] = 0

    for t in range(END_AGE - START_AGE):
        if t < 20:
            cond1 = P[:, t] >= D[:, t]*rate
            D[cond1, t + 1] = D[cond1, t] * (1 + rate) - P[cond1, t]
            cond2 = P[:, t] < D[:, t]*rate
            D[cond2, t + 1] = D[cond2, t] + (D[cond2, t]*rate - P[cond2, t]) / 2
        else:
            D[:, t] = 0

    for t in range(END_AGE - START_AGE):
        if t < 20:
            cond = D[:, t] < 0
            P[cond, t-1] = D[cond, t-1]
            D[cond, t] = 0

    adj_Y = Y - P
    return adj_Y


def exp_val_new(y, savings_incr, grid_w, v, N_SIM):

    COH = np.zeros((N_SIM, N_C))
    COH[:] = np.squeeze(savings_incr)
    COH += y[None].T

    COH[COH > grid_w[-1]] = grid_w[-1]
    COH[COH < grid_w[0]] = grid_w[0]

    spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

    v_w = np.zeros((N_SIM, N_C))
    for i in range(N_SIM):
        v_w[i, :] = spline(COH[i, :])

    ev = v_w.mean(axis=0)
    return ev[None].T




from scipy.interpolate import CubicSpline
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
from constants import unemp_frac, education_level, start_ages, RETIRE_AGE, MU, END_AGE, unemp_rate, N_C, ret_frac

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    if gamma == 1:
        return np.log(values)
    else:
        return values**(1-gamma) / (1-gamma)


def cal_income(coeffs, alt_deg):
    coeff_this_group = coeffs.loc[education_level[alt_deg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(start_ages[alt_deg], RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    return income


def read_input_data(income_fp, mortal_fp, id_fn, alt_deg):
    ids = pd.read_excel(id_fn)
    ids = ids[ids['AltDeg'] == education_level[alt_deg]]
    n_sim = len(ids)
    age_coeff_and_var = pd.ExcelFile(income_fp)
    # age coefficients
    age_coeff = pd.read_excel(age_coeff_and_var, sheet_name='Coefficients', index_col=0)

    # decomposed variance
    std = pd.read_excel(age_coeff_and_var, sheet_name='Variance', header=[1, 2], index_col=0)
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])

    # conditional survival probabilities
    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)

    return age_coeff, std, cond_prob, ids, n_sim


def adj_income_process(income, sigma_perm, sigma_tran, term, rho, n_sim, alt_deg):
    # generate random walk and normal r.v.
    np.random.seed(0)
    rn_perm = np.random.normal(MU, sigma_perm, (n_sim, RETIRE_AGE - start_ages[alt_deg] + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    np.random.seed(1)
    rn_tran = np.random.normal(MU, sigma_tran, (n_sim, RETIRE_AGE - start_ages[alt_deg] + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income)

    # - retirement
    ret_income_vec = ret_frac[alt_deg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # unemployment risk
    # generate bernoulli random variable
    np.random.seed(seed=2)
    p = 1 - unemp_rate[alt_deg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - start_ages[alt_deg] + 1, n_sim)).astype(float)
    r[r == 0] = unemp_frac[alt_deg]
    ones = np.ones((END_AGE - RETIRE_AGE, n_sim))
    bern = np.append(r, ones, axis=0)
    Y = np.multiply(inc_with_inc_risk, bern.T)

    # adjust income with ISA
    adj_Y = Y.copy()
    adj_Y[:, :term] *= rho
    return adj_Y


def exp_val_new(y, savings_incr, grid_w, v, N_SIM):

    COH = np.zeros((N_SIM, N_C))
    COH[:] = np.squeeze(savings_incr)
    COH += y[None].T

    COH[COH > grid_w[-1]] = grid_w[-1]
    COH[COH < grid_w[0]] = grid_w[0]

    spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

    # p = mp.Pool(processes=mp.cpu_count())
    # v_w = p.apply(spline, args=(COH,))
    # p.close()

    v_w = np.zeros((N_SIM, N_C))
    for i in range(N_SIM):
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
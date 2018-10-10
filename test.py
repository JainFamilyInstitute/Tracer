import os
import time
import numpy as np
import pandas as pd
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import *
from datetime import datetime

import multiprocessing as mp
import itertools

income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
isa_fn = 'Loop on term and rho.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
isa_fp = os.path.join(base_path, 'data', isa_fn)
ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)

income_bf_ret = cal_income(age_coeff)

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]


def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, gamma):
    term = int(param_pair[0])
    rho = param_pair[1]

    start = time.time()

    # adj income
    adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim)

    # get conditional survival probabilities
    cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
    cond_prob = cond_prob.values

    ###########################################################################
    #                    DP - read consumption functions                      #
    ###########################################################################
    date = "2018-8-24"
    c_func_fp = os.path.join(base_path, 'data', f'c_ISA_10_0.8514042436476386_2.0_2018-08-24.xlsx')
    c_func_df = pd.read_excel(c_func_fp)

    op = []
    for dec_l, dec_r in zip(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1) + 0.1):
        discount_array = DELTA**np.arange(adj_income.shape[1])
        discount_Y = np.multiply(adj_income, discount_array)
        npvs = np.sum(discount_Y, axis=1)
        allowed_rows = np.where(np.logical_and(npvs > np.percentile(npvs, 100*dec_l), npvs < np.percentile(npvs, 100*dec_r)))
        cur_decile = adj_income[allowed_rows]
        ###########################################################################
        #        CE - calculate consumption process & certainty equivalent        #
        ###########################################################################
        c_proc, _ = generate_consumption_process(cur_decile, c_func_df, cur_decile.shape[0])
        prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
        c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
        op.append(c_ce)
    df = pd.DataFrame(op)
    df.to_csv(os.path.join(base_path, 'results', 'ces.csv'))

    print(
        f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
    print(f"------ {time.time() - start} seconds ------")

    return term, rho, gamma, c_ce



param_pair = [10, 0.8514042436476386]
run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000, 2.0)
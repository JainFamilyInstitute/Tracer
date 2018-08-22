import os
import time
import numpy as np
import pandas as pd
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import *

import multiprocessing as mp
import itertools


def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran,  surv_prob, base_path, n_sim, gamma):
    principal = param_pair[0]
    ppt_bar = param_pair[1]

    start = time.time()

    # adj income
    adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)

    # get conditional survival probabilities
    cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
    cond_prob = cond_prob.values

    ###########################################################################
    #                  DP - generate consumption functions                    #
    ###########################################################################
    c_func_fp = os.path.join(base_path, 'results', f'c function_DEBT_{ppt_bar}.xlsx')
    v_func_fp = os.path.join(base_path, 'results', f'v function_DEBT_{ppt_bar}.xlsx')
    # shortcut:
    # c_func_df = pd.read_excel(c_func_fp)
    # v_func_df = pd.read_excel(v_func_fp)
    c_func_df, v_func_df = dp_solver(adj_income, cond_prob, gamma, n_sim)
    c_func_df.to_excel(c_func_fp)
    v_func_df.to_excel(v_func_fp)
    ###########################################################################
    #        CE - calculate consumption process & certainty equivalent        #
    ###########################################################################
    #adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran)
    c_proc, _ = generate_consumption_process(adj_income, c_func_df, n_sim)

    prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values

    c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)

    ##Expanding Factor
    print(f'########## Gamma: {ppt_bar} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
    print(f"------ {time.time() - start} seconds ------")

    #print(f'########## Gamma: {ppt_bar} | CE: {c_ce} | {time.time() - start} seconds ##########')
    return principal, ppt_bar, gamma, c_ce

def main(version, n_sim, gamma):
    assert version == 'DEBT'
    start_time = time.time()

    ###########################################################################
    #                      Setup - file path & raw data                       #
    ###########################################################################
    # set file path
    income_fn = 'age_coefficients_and_var.xlsx'
    surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
    loan_fn = 'Loop on Principal for Loan.xlsx'
    base_path = os.path.dirname(__file__)
    income_fp = os.path.join(base_path, 'data', income_fn)
    mortal_fp = os.path.join(base_path, 'data', surviv_fn)
    loan_fp = os.path.join(base_path, 'data', loan_fn)
    ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

    # read raw data
    age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)


    ###########################################################################
    #              Setup - income process & std & survival prob               #
    ###########################################################################
    income_bf_ret = cal_income(age_coeff)
    # income_ret = income_bf_ret[-1]

    # get std
    sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
    sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]

    loan_params = pd.read_excel(loan_fp)
    loan_params = loan_params[["New Principal", "ppt-bar"]].copy()

    param_pair = list(loan_params.values)
    fixed_args = [[x] for x in [income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim]]

    if isinstance(gamma,float):
        gamma = [gamma]

    search_args = list(itertools.product(param_pair, *fixed_args, gamma))
    with mp.Pool(processes=mp.cpu_count()) as p:
        c_ce = p.starmap(run_model, search_args)

    # c_ce = np.zeros((len(gamma_arr), 2))
    # for i in range(len(gamma_arr)):
    #     c_ce[i, 0], c_ce[i, 1] = run_model(gamma_arr[i])

    c_ce_df = pd.DataFrame(c_ce, columns=['principal', 'ppt_bar', 'gamma', 'Consumption CE'])
    c_ce_df.to_excel(ce_fp)


    # Params check
    print("--- %s seconds ---" % (time.time() - start_time))
    print('AltDeg: ', AltDeg)
    print('permanent shock: ', sigma_perm)
    print('transitory shock: ', sigma_tran)
    print('lambda: ', ret_frac[AltDeg])
    print('theta: ',  unemp_frac[AltDeg])
    print('pi: ', unempl_rate[AltDeg])
    print('W0: ', INIT_WEALTH)

import os
import time
import numpy as np
import pandas as pd
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process, cal_ce_agent
from constants import *
from datetime import datetime
import glob

import multiprocessing as mp
import itertools

income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
loan_fn = 'Loop on Principal for Loan.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)
loan_fp = os.path.join(base_path, 'data', loan_fn)

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)

income_bf_ret = cal_income(age_coeff)

# get std
sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]


# # hsg
# def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     principal = param_pair[0]
#     ppt_bar = param_pair[1]
#
#     start = time.time()
#
#     # adj income
#     adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)
#
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_High*'))
#
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         c_proc, _ = generate_consumption_process(adj_income, c_func_df, adj_income.shape[0])
#         prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#         ce = cal_ce_agent(prob, c_proc, gamma)
#
#         print(f'########## Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#
#         col_names = [str(x + START_AGE) for x in range(END_AGE - START_AGE + 1)]
#         df = pd.DataFrame(adj_income, columns=col_names)
#         df['CE'] = ce
#         df.to_csv(os.path.join(base_path, 'results', f'hsg_inc_CEs_Gamma{gamma}.csv'))
#
#
#
# param_pair = [None, None]
# run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# # cg
# def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     principal = param_pair[0]
#     ppt_bar = param_pair[1]
#
#     start = time.time()
#
#     adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)
#
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_Coll*'))
#
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         c_proc, _ = generate_consumption_process(adj_income, c_func_df, adj_income.shape[0])
#         prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#         ce = cal_ce_agent(prob, c_proc, gamma)
#
#         print(f'########## Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#
#         col_names = [str(x + START_AGE) for x in range(END_AGE - START_AGE + 1)]
#         df = pd.DataFrame(adj_income, columns=col_names)
#         df['CE'] = ce
#         df.to_csv(os.path.join(base_path, 'results', f'cg_inc_CEs_Gamma{gamma}.csv'))
#
#
# param_pair = [None, None]
# run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# debt
debt_dict = {
    805: 5000,
    1610: 10000,
    2416: 15000,
    3221: 20000,
    4027: 25000,
    4832: 30000,
    5638: 35000,
    6443: 40000,
    7249: 45000,
    8054: 50000,
    8859: 55000,
    9665: 60000,
    10470: 65000,
    11276: 70000
}


def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):

    start = time.time()

    cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_DEBT*'))

    for fp in cfunc_fps:
        c_func_df = pd.read_excel(fp)
        gamma = float(fp.split('.')[-3][-1])
        ppt_bar = float(fp.split('/')[-1].split('_')[2])
        principal = debt_dict[int(ppt_bar)]

        adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)

        c_proc, _ = generate_consumption_process(adj_income, c_func_df, adj_income.shape[0])
        prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
        ce = cal_ce_agent(prob, c_proc, gamma)

        print(f'########## ppt_bar: {ppt_bar} | principal: {principal} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} ##########')
        print(f"------ {time.time() - start} seconds ------")

        col_names = [str(x + START_AGE) for x in range(END_AGE - START_AGE + 1)]
        df = pd.DataFrame(adj_income, columns=col_names)
        df['CE'] = ce
        df.to_csv(os.path.join(base_path, 'results', f'DEBT_inc_CEs_pptbar{ppt_bar}_Gamma{gamma}.csv'))

run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)



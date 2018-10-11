import os
import time
import numpy as np
import pandas as pd
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
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
#     # get conditional survival probabilities
#     cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
#     cond_prob = cond_prob.values
#
#     ###########################################################################
#     #                    DP - read consumption functions                      #
#     ###########################################################################
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_High*'))
#
#     op = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         single_op = []
#         for dec_l, dec_r in zip(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1) + 0.1):
#             discount_array = DELTA**np.arange(adj_income.shape[1])
#             discount_Y = np.multiply(adj_income, discount_array)
#             npvs = np.sum(discount_Y, axis=1)
#             allowed_rows = np.where(np.logical_and(npvs >= np.percentile(npvs, 100*dec_l), npvs < np.percentile(npvs, 100*dec_r)))
#             cur_decile = adj_income[allowed_rows]
#             ###########################################################################
#             #        CE - calculate consumption process & certainty equivalent        #
#             ###########################################################################
#             c_proc, _ = generate_consumption_process(cur_decile, c_func_df, cur_decile.shape[0])
#             prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#             c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
#             single_op.append(c_ce)
#
#         print(f'########## Gamma: {ppt_bar} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#         op.append(single_op)
#     df = pd.DataFrame(op)
#     df.to_csv(os.path.join(base_path, 'results', 'hsg_CEs.csv'))
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
#     # adj income
#     adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)
#
#     # get conditional survival probabilities
#     cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
#     cond_prob = cond_prob.values
#
#     ###########################################################################
#     #                    DP - read consumption functions                      #
#     ###########################################################################
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_Coll*'))
#
#     op = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         single_op = []
#         for dec_l, dec_r in zip(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1) + 0.1):
#             discount_array = DELTA**np.arange(adj_income.shape[1])
#             discount_Y = np.multiply(adj_income, discount_array)
#             npvs = np.sum(discount_Y, axis=1)
#             allowed_rows = np.where(np.logical_and(npvs >= np.percentile(npvs, 100*dec_l), npvs < np.percentile(npvs, 100*dec_r)))
#             cur_decile = adj_income[allowed_rows]
#             ###########################################################################
#             #        CE - calculate consumption process & certainty equivalent        #
#             ###########################################################################
#             c_proc, _ = generate_consumption_process(cur_decile, c_func_df, cur_decile.shape[0])
#             prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#             c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
#             single_op.append(c_ce)
#
#         print(f'########## Gamma: {ppt_bar} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#         op.append(single_op)
#     df = pd.DataFrame(op)
#     df.to_csv(os.path.join(base_path, 'results', 'cg_CEs.csv'))
#
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

    op = []
    idx_names = []
    for fp in cfunc_fps:
        c_func_df = pd.read_excel(fp)
        gamma = float(fp.split('.')[-3][-1])
        ppt_bar = float(fp.split('/')[-1].split('_')[2])
        principal = debt_dict[int(ppt_bar)]
        idx_names.append(str(ppt_bar) + '_' + str(gamma))

        adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)

        single_op = []
        for dec_l, dec_r in zip(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1) + 0.1):
            discount_array = DELTA**np.arange(adj_income.shape[1])
            discount_Y = np.multiply(adj_income, discount_array)
            npvs = np.sum(discount_Y, axis=1)
            allowed_rows = np.where(np.logical_and(npvs >= np.percentile(npvs, 100*dec_l), npvs < np.percentile(npvs, 100*dec_r)))
            cur_decile = adj_income[allowed_rows]
            ###########################################################################
            #        CE - calculate consumption process & certainty equivalent        #
            ###########################################################################
            c_proc, _ = generate_consumption_process(cur_decile, c_func_df, cur_decile.shape[0])
            prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
            c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
            single_op.append(c_ce)

        print(f'########## ppt_bar: {ppt_bar} | principal: {principal} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
        print(f"------ {time.time() - start} seconds ------")
        op.append(single_op)
    df = pd.DataFrame(op, index=idx_names)
    df.to_csv(os.path.join(base_path, 'results', 'debt_CEs.csv'))


run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# plan
# 1. hsg - done
# 2. cg - done
# 3. debt

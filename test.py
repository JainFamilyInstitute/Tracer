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


# debt for quantile
# output: For 5k, 30k, 70k, for each quantile,
#         average income, average payments, average consumption, average utility
debt_dict = {
    805: 5000,
    4832: 30000,
    11276: 70000,
}


def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):

    start = time.time()

    cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_Debt*'))

    op = []
    idx_names = []
    for fp in cfunc_fps:
        c_func_df = pd.read_excel(fp)
        gamma = float(fp.split('.')[-3][-1])
        ppt_bar = float(fp.split('/')[-1].split('_')[2])

        iterables = [[1, 2, 3, 4],
                     ['Income', 'Payment', 'Consumption', 'Utility']]
        index = pd.MultiIndex.from_product(iterables, names=['Quartile', 'Category'])
        column = np.arange(22, 101)
        ipcu_df = pd.DataFrame(index=index, columns=column)

        if int(ppt_bar) in debt_dict:
            principal = debt_dict[int(ppt_bar)]
            idx_names.append(str(ppt_bar) + '_' + str(gamma))

            adj_income, income, payments = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)

            single_op = []
            for l, r, i in zip(np.arange(0, 1, 0.25), np.arange(0, 1, 0.25) + 0.25, np.arange(4) + 1):
                discount_array = DELTA**np.arange(adj_income.shape[1])
                discount_Y = np.multiply(adj_income, discount_array)
                npvs = np.sum(discount_Y, axis=1)
                allowed_rows = np.where(np.logical_and(npvs >= np.quantile(npvs, l), npvs < np.quantile(npvs, r)))
                cur_intvl = adj_income[allowed_rows]

                # op: Income, Payment, Adjusted Income
                q_ave_inc = np.mean(income[allowed_rows], axis=0)
                q_ave_pmt = np.mean(payments[allowed_rows], axis=0)

                c_proc, _ = generate_consumption_process(cur_intvl, c_func_df, cur_intvl.shape[0])
                # op: Consumption
                q_ave_c = np.mean(c_proc, axis=0)

                prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
                c_ce, _, util = cal_certainty_equi(prob, c_proc, gamma)
                # op: Utility
                q_ave_util = np.mean(util, axis=0)
                single_op.append(c_ce)
                ipcu_df.loc[i] = np.array([q_ave_inc, q_ave_pmt, q_ave_c, q_ave_util])

            print(f'########## ppt_bar: {ppt_bar} | principal: {principal} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
            print(f"------ {time.time() - start} seconds ------")
            op.append(single_op)

            ipcu_df.to_csv(os.path.join(base_path, 'results', f'debt_quantile_IPCU_{ppt_bar}_gamma_{gamma}.csv'))
    df = pd.DataFrame(op, index=idx_names)
    df.to_csv(os.path.join(base_path, 'results', 'debt_quantile_CEs.csv'))


run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# # hsg for quantile
# def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     principal = param_pair[0]
#     ppt_bar = param_pair[1]
#
#     start = time.time()
#
#     income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)
#
#     ###########################################################################
#     #                    DP - read consumption functions                      #
#     ###########################################################################
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_HSG*'))
#
#     op = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         iterables = [[1, 2, 3, 4],
#                      ['Income', 'Consumption', 'Utility']]
#         index = pd.MultiIndex.from_product(iterables, names=['Quartile', 'Category'])
#         column = np.arange(18, 101)
#         ipcu_df = pd.DataFrame(index=index, columns=column)
#
#         single_op = []
#         for l, r, i in zip(np.arange(0, 1, 0.25), np.arange(0, 1, 0.25) + 0.25, np.arange(4) + 1):
#             discount_array = DELTA ** np.arange(income.shape[1])
#             discount_Y = np.multiply(income, discount_array)
#             npvs = np.sum(discount_Y, axis=1)
#             allowed_rows = np.where(np.logical_and(npvs >= np.quantile(npvs, l), npvs < np.quantile(npvs, r)))
#             cur_intvl = income[allowed_rows]
#
#             # op: Income, Payment, Adjusted Income
#             q_ave_inc = np.mean(income[allowed_rows], axis=0)
#
#             c_proc, _ = generate_consumption_process(cur_intvl, c_func_df, cur_intvl.shape[0])
#             # op: Consumption
#             q_ave_c = np.mean(c_proc, axis=0)
#
#             prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#             c_ce, _, util = cal_certainty_equi(prob, c_proc, gamma)
#             # op: Utility
#             q_ave_util = np.mean(util, axis=0)
#             single_op.append(c_ce)
#             ipcu_df.loc[i] = np.array([q_ave_inc, q_ave_c, q_ave_util])
#
#         print(f'########## Gamma: {ppt_bar} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#         op.append(single_op)
#         ipcu_df.to_csv(os.path.join(base_path, 'results', f'hsg_quantile_IPCU_gamma_{gamma}.csv'))
#     df = pd.DataFrame(op)
#     df.to_csv(os.path.join(base_path, 'results', 'hsg_quantile_CEs.csv'))
#
#
# param_pair = [None, None]
# run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# # cg for quantile
# def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     principal = param_pair[0]
#     ppt_bar = param_pair[1]
#
#     start = time.time()
#
#     income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, ppt_bar, n_sim)
#
#     ###########################################################################
#     #                    DP - read consumption functions                      #
#     ###########################################################################
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_CG*'))
#
#     op = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         gamma = float(fp.split('.')[-3][-1])
#
#         iterables = [[1, 2, 3, 4],
#                      ['Income', 'Consumption', 'Utility']]
#         index = pd.MultiIndex.from_product(iterables, names=['Quartile', 'Category'])
#         column = np.arange(22, 101)
#         ipcu_df = pd.DataFrame(index=index, columns=column)
#
#         single_op = []
#         for l, r, i in zip(np.arange(0, 1, 0.25), np.arange(0, 1, 0.25) + 0.25, np.arange(4) + 1):
#             discount_array = DELTA ** np.arange(income.shape[1])
#             discount_Y = np.multiply(income, discount_array)
#             npvs = np.sum(discount_Y, axis=1)
#             allowed_rows = np.where(np.logical_and(npvs >= np.quantile(npvs, l), npvs < np.quantile(npvs, r)))
#             cur_intvl = income[allowed_rows]
#
#             # op: Income, Payment, Adjusted Income
#             q_ave_inc = np.mean(income[allowed_rows], axis=0)
#
#             c_proc, _ = generate_consumption_process(cur_intvl, c_func_df, cur_intvl.shape[0])
#             # op: Consumption
#             q_ave_c = np.mean(c_proc, axis=0)
#
#             prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#             c_ce, _, util = cal_certainty_equi(prob, c_proc, gamma)
#             # op: Utility
#             q_ave_util = np.mean(util, axis=0)
#             single_op.append(c_ce)
#             ipcu_df.loc[i] = np.array([q_ave_inc, q_ave_c, q_ave_util])
#
#         print(f'########## Gamma: {ppt_bar} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#         op.append(single_op)
#         ipcu_df.to_csv(os.path.join(base_path, 'results', f'cg_quantile_IPCU_gamma_{gamma}.csv'))
#     df = pd.DataFrame(op)
#     df.to_csv(os.path.join(base_path, 'results', 'cg_quantile_CEs.csv'))
#
#
# param_pair = [None, None]
# run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)

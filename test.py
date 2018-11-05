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

# # by agent
# def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     term = 10
#
#     start = time.time()
#
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_ISA_*'))
#
#     col_names = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         rho = float(fp.split('/')[-1].split('_')[3])
#         gamma = float(fp.split('/')[-1].split('_')[4])
#         col_names.append(str(rho)+'_'+str(gamma))
#
#         adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim)
#
#         c_proc, _ = generate_consumption_process(adj_income, c_func_df, adj_income.shape[0])
#         prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#         ce = cal_ce_agent(prob, c_proc, gamma)
#
#         print(
#             f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#
#         col_names = [str(x + START_AGE) for x in range(END_AGE - START_AGE + 1)]
#         df = pd.DataFrame(adj_income, columns=col_names)
#         df['CE'] = ce
#         df.to_csv(os.path.join(base_path, 'results', f'ISA_inc_CEs_rho{rho}_Gamma{gamma}.csv'))
#
#
# run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)


# # output utils
# def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
#     term = 10
#
#     start = time.time()
#
#     cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_ISA_*'))
#
#     col_names = []
#     for fp in cfunc_fps:
#         c_func_df = pd.read_excel(fp)
#         rho = float(fp.split('/')[-1].split('_')[3])
#         gamma = float(fp.split('/')[-1].split('_')[4])
#         col_names.append(str(rho)+'_'+str(gamma))
#
#         adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim)
#
#         c_proc, _ = generate_consumption_process(adj_income, c_func_df, adj_income.shape[0])
#         prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
#         ce, util = cal_ce_agent(prob, c_proc, gamma)
#
#         print(
#             f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} ##########')
#         print(f"------ {time.time() - start} seconds ------")
#
#         col_names = [str(x + START_AGE) for x in range(END_AGE - START_AGE + 1)]
#         df = pd.DataFrame(util, columns=['Utility'])
#         df['CE'] = ce
#         df.to_csv(os.path.join(base_path, 'results', f'ISA_Utils_CEs_rho{rho}_Gamma{gamma}.csv'))
#
#
# run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)



def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
    term = 10

    start = time.time()

    cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_ISA_*_2018-10-31.xlsx'))

    op = []
    idx_names = []
    for fp in cfunc_fps:
        c_func_df = pd.read_excel(fp)
        rho = float(fp.split('/')[-1].split('_')[3])
        gamma = float(fp.split('/')[-1].split('_')[4])

        iterables = [[1, 2, 3, 4],
                     ['Income', 'Payment', 'Consumption', 'Utility']]
        index = pd.MultiIndex.from_product(iterables, names=['Quartile', 'Category'])
        column = np.arange(22, 101)
        ipcu_df = pd.DataFrame(index=index, columns=column)

        idx_names.append(str(rho)+'_'+str(gamma))

        adj_income, income, payments = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim)

        single_op = []
        for l, r, i in zip(np.arange(0, 1, 0.25), np.arange(0, 1, 0.25) + 0.25, np.arange(4) + 1):
            discount_array = DELTA**np.arange(adj_income.shape[1])
            discount_Y = np.multiply(adj_income, discount_array)
            npvs = np.sum(discount_Y, axis=1)
            allowed_rows = np.where(np.logical_and(npvs > np.quantile(npvs, l), npvs < np.quantile(npvs, r)))
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

        print(
            f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
        print(f"------ {time.time() - start} seconds ------")
        op.append(single_op)

        ipcu_df.to_csv(os.path.join(base_path, 'results', f'ISA_quantile_IPCU_{rho}_gamma_{gamma}.csv'))
    df = pd.DataFrame(op, index=idx_names)
    df.to_csv(os.path.join(base_path, 'results', 'ISA_quantile_CEs.csv'))


run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)

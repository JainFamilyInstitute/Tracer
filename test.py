import os
from constants import *
from functions import *
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process


income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob Feb 16.xlsx'
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', income_fn)
mortal_fp = os.path.join(base_path, 'data', surviv_fn)

# read raw data
age_coeff, std, surv_prob = read_input_data(income_fp, mortal_fp)

income_bf_ret = cal_income(age_coeff)

sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[AltDeg]]
sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[AltDeg]]

# INIT_DEBT = 33942.72008
#
# adj_income, payment, debt = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, INIT_DEBT, 10000)
#
# # inc_proc = pd.DataFrame(adj_income)
# # inc_fp = os.path.join(base_path, 'results', 'inc_proc_cg.csv')
# # inc_proc.to_csv(inc_fp)
# #
# # p_proc = pd.DataFrame(payment)
# # p_fp = os.path.join(base_path, 'results', 'p_proc_cg.csv')
# # p_proc.to_csv(p_fp)
# #
# # debt_proc = pd.DataFrame(debt)
# # debt_fp = os.path.join(base_path, 'results', 'debt_proc_cg.csv')
# # debt_proc.to_csv(debt_fp)
#
# # get conditional survival probabilities
# cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
# cond_prob = cond_prob.values
#
# gamma = 2
# c_func_df, v_func_df = dp_solver(adj_income, cond_prob, gamma, 10000)
# c_proc, _ = generate_consumption_process(adj_income, c_func_df, 10000)
#
# prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
# c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
# print(c_ce)



####################################################################################################################
import glob
import time


# for Quartile IPCU
def run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim):
    term = 10

    start = time.time()

    cfunc_fps = glob.glob(os.path.join(base_path, 'data', 'c_IDR_*'))

    op = []
    idx_names = []
    for fp in cfunc_fps:
        c_func_df = pd.read_excel(fp)
        principal = float(fp.split('/')[-1].split('_')[2])
        gamma = float(fp.split('/')[-1].split('_')[3])

        iterables = [[1, 2, 3, 4],
                     ['Income', 'Payment', 'Consumption', 'Utility']]
        index = pd.MultiIndex.from_product(iterables, names=['Quartile', 'Category'])
        column = np.arange(22, 101)
        ipcu_df = pd.DataFrame(index=index, columns=column)

        idx_names.append(str(principal)+'_'+str(gamma))

        adj_income, payment, income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, principal, n_sim)

        single_op = []
        for l, r, i in zip(np.arange(0, 1, 0.25), np.arange(0, 1, 0.25) + 0.25, np.arange(4) + 1):
            discount_array = DELTA**np.arange(adj_income.shape[1])
            discount_Y = np.multiply(adj_income, discount_array)
            npvs = np.sum(discount_Y, axis=1)
            allowed_rows = np.where(np.logical_and(npvs > np.quantile(npvs, l), npvs < np.quantile(npvs, r)))
            cur_intvl = adj_income[allowed_rows]

            # op: Income, Payment, Adjusted Income
            q_ave_inc = np.mean(income[allowed_rows], axis=0)
            q_ave_pmt = np.mean(payment[allowed_rows], axis=0)

            c_proc, _ = generate_consumption_process(cur_intvl, c_func_df, cur_intvl.shape[0])
            # op: Consumption
            q_ave_c = np.mean(c_proc, axis=0)

            prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
            c_ce, util = cal_certainty_equi(prob, c_proc, gamma)
            # op: Utility
            q_ave_util = np.mean(util, axis=0)

            single_op.append(c_ce)
            ipcu_df.loc[i] = np.array([q_ave_inc, q_ave_pmt, q_ave_c, q_ave_util])

        print(
            f'########## Term: {term} | Principal: {principal:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
        print(f"------ {time.time() - start} seconds ------")
        op.append(single_op)

        ipcu_df.to_csv(os.path.join(base_path, 'results', f'IDR_quantile_IPCU_{principal}_gamma_{gamma}.csv'))
    df = pd.DataFrame(op, index=idx_names)
    df.to_csv(os.path.join(base_path, 'results', 'IDR_quantile_CEs.csv'))


run_model(income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, 10000)

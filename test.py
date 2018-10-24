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

INIT_DEBT = 33942.72008
PL = 1.5 * 10830

adj_income, payment, debt = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, INIT_DEBT, PL, 10000)

# inc_proc = pd.DataFrame(adj_income)
# inc_fp = os.path.join(base_path, 'results', 'inc_proc_cg.csv')
# inc_proc.to_csv(inc_fp)
#
# p_proc = pd.DataFrame(payment)
# p_fp = os.path.join(base_path, 'results', 'p_proc_cg.csv')
# p_proc.to_csv(p_fp)
#
# debt_proc = pd.DataFrame(debt)
# debt_fp = os.path.join(base_path, 'results', 'debt_proc_cg.csv')
# debt_proc.to_csv(debt_fp)

# get conditional survival probabilities
cond_prob = surv_prob.loc[START_AGE:END_AGE - 1, 'CSP']  # 22:99
cond_prob = cond_prob.values

gamma = 2
c_func_df, v_func_df = dp_solver(adj_income, cond_prob, gamma, 10000)
c_proc, _ = generate_consumption_process(adj_income, c_func_df, 10000)

prob = surv_prob.loc[START_AGE:END_AGE, 'CSP'].cumprod().values
c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
print(c_ce)

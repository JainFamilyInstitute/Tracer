import os
from constants import *
from functions import *


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

adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, 0, 0, 10000)
inc_proc = pd.DataFrame(adj_income)

inc_fp = os.path.join(base_path, 'results', 'inc_proc_cg.csv')
inc_proc.to_csv(inc_fp)
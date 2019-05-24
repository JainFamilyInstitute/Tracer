import os
import time
import pandas as pd
from functions import adj_income_process, read_input_data, cal_income
from dp import dp_solver
from cal_ce import cal_certainty_equi, generate_consumption_process
from constants import start_ages, END_AGE, gamma_exp_frac, education_level, ret_frac, unemp_rate, unemp_frac, INIT_WEALTH
from datetime import datetime

import multiprocessing as mp
import itertools


#SIDHYA CHANGE
def run_model(param_pair, income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg, gamma):
    term = int(param_pair[0])
    rho = param_pair[1]
    
    start = time.time()

    # adj income
    #SIDHYA CHANGE
    adj_income = adj_income_process(income_bf_ret, sigma_perm, sigma_tran, term, rho, n_sim, alt_deg)

    # get conditional survival probabilities
    cond_prob = surv_prob.loc[start_ages[alt_deg]:END_AGE - 1, 'CSP']  # 22:99
    cond_prob = cond_prob.values

    ###########################################################################
    #                  DP - generate consumption functions                    #
    ###########################################################################
    today = datetime.now().date()
    c_func_fp = os.path.join(base_path, 'results', f'c_ISA_{term}_{rho}_{gamma}_{today}.xlsx')
    v_func_fp = os.path.join(base_path, 'results', f'v_ISA_{term}_{rho}_{gamma}_{today}.xlsx')
    # shortcut:
    # c_func_df = pd.read_excel(c_func_fp)
    # v_func_df = pd.read_excel(v_func_fp)
    c_func_df, v_func_df = dp_solver(adj_income, cond_prob, gamma, n_sim, alt_deg)
    c_func_df.to_excel(c_func_fp)
    v_func_df.to_excel(v_func_fp)
    ###########################################################################
    #        CE - calculate consumption process & certainty equivalent        #
    ###########################################################################
    c_proc, _ = generate_consumption_process(adj_income, c_func_df, n_sim)

    prob = surv_prob.loc[start_ages[alt_deg]:END_AGE, 'CSP'].cumprod().values

    c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
    #SIDHYA CHANGE
    ##Expanding Factor
    print(f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
    print(f"------ {time.time() - start} seconds ------")
    return term, rho, gamma, c_ce


def main(version, id_fn, alt_deg, gamma):
    assert version == 'ISA_MC_from_ids'
    start_time = time.time()

    ###########################################################################
    #                      Setup - file path & raw data                       #
    ###########################################################################
    income_fn = 'age_coefficients_and_var_May152019.xlsx'
    surviv_fn = 'Conditional Survival Prob_May152019.xlsx'
    isa_fn = 'Loop on term and rho.xlsx'
    base_path = os.path.dirname(__file__)
    income_fp = os.path.join(base_path, 'data', income_fn)
    mortal_fp = os.path.join(base_path, 'data', surviv_fn)
    isa_fp = os.path.join(base_path, 'data', isa_fn)
    ce_fp = os.path.join(base_path, 'results', 'ce.xlsx')

    # read raw data
    age_coeff, std, surv_prob, ids, n_sim = read_input_data(income_fp, mortal_fp, id_fn, alt_deg)


    ###########################################################################
    #              Setup - income process & std & survival prob               #
    ###########################################################################
    income_bf_ret = cal_income(age_coeff, alt_deg)

    # get std
    sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[alt_deg]]
    sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[alt_deg]]
    #SIDHYA CHANGE
    isa_params = pd.read_excel(isa_fp)
    isa_params = isa_params[["Term", "1-rho"]].copy()
    #SIDHYA CHANGE
    param_pair = list(isa_params.values)
    fixed_args = [[x] for x in [income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg]]

    if isinstance(gamma, float):
        gamma = [gamma]

    search_args = list(itertools.product(param_pair, *fixed_args, gamma))

    # print(param_pair)
    # export_incomes(param_pair[0], income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg, gamma[0])

    with mp.Pool(processes=mp.cpu_count()) as p:
        c_ce = p.starmap(run_model, search_args)

    c_ce_df = pd.DataFrame(c_ce, columns=['Term', 'Rho', 'gamma', 'Consumption CE'])
    c_ce_df.to_excel(ce_fp)

    # Params check
    print("--- %s seconds ---" % (time.time() - start_time))
    print('AltDeg: ', alt_deg)
    print('n_sim ', n_sim )
    print('permanent shock: ', sigma_perm)
    print('transitory shock: ', sigma_tran)
    print('lambda: ', ret_frac[alt_deg])
    print('theta: ',  unemp_frac[alt_deg])
    print('pi: ', unemp_rate[alt_deg])
    print('W0: ', INIT_WEALTH)

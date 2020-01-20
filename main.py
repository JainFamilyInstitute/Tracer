import os
import time
import pandas as pd
from file_handlers import  read_ids_for_alt_deg, read_age_coeffs, read_variance, read_survival
from income_process import adj_income_process, cal_income
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
    if alt_deg == 3:
        rho = rho/2
    else:
        assert alt_deg == 4
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
    c_func_df, v_func_df = dp_solver(adj_income=adj_income, cond_prob=cond_prob, gamma=gamma, n_sim=n_sim, alt_deg=alt_deg)
    c_func_df.to_excel(c_func_fp)
    v_func_df.to_excel(v_func_fp)
    ###########################################################################
    #        CE - calculate consumption process & certainty equivalent        #
    ###########################################################################
    c_proc, _ = generate_consumption_process(adj_income, c_func_df, n_sim, start_age)

    prob = surv_prob.loc[start_ages[alt_deg]:END_AGE, 'CSP'].cumprod().values

    c_ce, _ = cal_certainty_equi(prob, c_proc, gamma)
    #SIDHYA CHANGE
    ##Expanding Factor
    print(f'########## Term: {term} | Rho: {rho:.2f} | Gamma: {gamma} | Exp_Frac: {gamma_exp_frac[gamma]} | CE: {c_ce:.2f} ##########')
    print(f"------ {time.time() - start} seconds ------")
    return term, rho, gamma, c_ce


def main(alt_degs, gammas):
    start_time = time.time()

    ###########################################################################
    #                      Setup - file path & raw data                       #
    ###########################################################################

    # read raw data
    age_coeffs = read_age_coeffs()
    std = read_variance()
    surv_prob = read_survival()

    # TODO alt_deg => alt_degs, separate methods for isa, debt
    ids = {alt_deg : read_ids_for_alt_deg(alt_deg) for alt_deg in alt_degs}
    
    ###########################################################################
    #              Setup - income process & std & survival prob               #
    ###########################################################################
    income_bf_ret = cal_income(age_coeffs, alt_deg)

    # get std
    sigma_perm = std.loc['sigma_permanent', 'Labor Income Only'][education_level[alt_deg]]
    sigma_tran = std.loc['sigma_transitory', 'Labor Income Only'][education_level[alt_deg]]
    
    ## TODO isa_params
    isa_params = pd.read_excel(isa_fp)
    isa_params = isa_params[["Term", "rho"]].copy()
    #SIDHYA CHANGE
    param_pair = list(isa_params.values)
    fixed_args = [[x] for x in [income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg]]

    if isinstance(gammas, float):
        gammas = [gammas]

    search_args = list(itertools.product(param_pair, *fixed_args, gammas))

    # print(param_pair)
    # export_incomes(param_pair[0], income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg, gammas[0])
    print('AltDeg: ', alt_deg)
    print('n_sim ', n_sim)
    print('permanent shock: ', sigma_perm)
    print('transitory shock: ', sigma_tran)
    print('lambda: ', ret_frac[alt_deg])
    print('theta: ', unemp_frac[alt_deg])
    print('pi: ', unemp_rate[alt_deg])
    print('W0: ', INIT_WEALTH)
    print('term: ', param_pair[0])
    print('rho: ', param_pair[1])
    print('[income_bf_ret, sigma_perm, sigma_tran, surv_prob, base_path, n_sim, alt_deg]: ', fixed_args)

    with mp.Pool(processes=mp.cpu_count()) as p:
        c_ce = p.starmap(run_model, search_args)

    c_ce_df = pd.DataFrame(c_ce, columns=['Term', 'Rho', 'gamma', 'Consumption CE'])
    c_ce_df.to_excel(ce_fp)

    # Params check
    print("--- %s seconds ---" % (time.time() - start_time))


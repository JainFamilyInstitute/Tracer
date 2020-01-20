###########################################################################
#                            Set Constants                                #
###########################################################################
END_AGE = 100             
RETIRE_AGE = 65                 # retirement age
N_W = 501
LOWER_BOUND_W = 1               # lower bound of wealth
UPPER_BOUND_W = 15000000        # upper bound of wealth
N_C = 1501
LOWER_BOUND_C = 0
R = 0.02                        # risk-free rate
DELTA = 0.99                    # discount factor
MU = 0                          # expectation of income shocks
INIT_WEALTH = 0                 #
ISA_POVERTY_MULTIPLIER = 2.0    # poverty multiplier for ISA lower threshold
IDR_POVERTY_MULTIPLIER = 1.5    # poverty multiplier for IDR
POVERTY_LEVEL = 10830           # poverty level
TERM_EXT = 5                    # number of term extensions for ISA
NOMINAL_CAP_MULTIPLIER = 2.5    # nominal cap multiplier for ISA
run_dp = True

# file names
income_fn = 'age_coefficients_and_var.xlsx'
surviv_fn = 'Conditional Survival Prob.xlsx'
isa_loop_fn = 'Loop on term and rho.xlsx'
debt_loop_fn = 'Loop on Principal for Loan.xlsx'
id_fn = "Mixture_IDS.xls"
ce_fn = 'ce.xlsx'

start_ages = {
    2: 18,
    3: 20,
    4: 22
}

# alt_deg to education level map
education_level = {
    2: 'High School Graduates',
    3: 'Some College',
    4: 'College Graduates'
}

# replacement rate of retirement earnings (lambda) 
ret_frac = {
    2: 0.5790,
    3: 0.5596,
    4: 0.4650,
}

# replacement rate of unemployment earnings (theta)
unemp_frac = {
    2: 0.6646,
    3: 0.6157,
    4: 0.5285 
}

# probability of suffering an unemployed spell (pi)
unemp_rate = {
    2: 0.1431,
    3: 0.1132,
    4: 0.0703
}

# expanding factor
gamma_exp_frac = {
    1: 3,
    2: 3,
    3: 4,
    4: 5,
    5: 5,
    6: 5,
    7: 5,
    8: 5,
}

###########################################################################
#                            Set Constants                                #
###########################################################################
#                  Sidhya Edit for new Rerun May 15 2019                  #

START_AGE = 22            #Set to 18 if High School Graduate, 20 if Some College and 22 if College Graduate
END_AGE = 100             #
RETIRE_AGE = 65           # retirement age
N_W = 501
LOWER_BOUND_W = 1         # lower bound of wealth
UPPER_BOUND_W = 15000000    # upper bound of wealth
EXPAND_FAC = 5
N_C = 1501
LOWER_BOUND_C = 0.1
# GAMMA = 6.5                 # risk preference parameter
R = 0.02                  # risk-free rate
DELTA = 0.99              # discount factor
MU = 0                    # expectation of income shocks
N_SIM = 100000            # number of draws
INIT_WEALTH = 0

AltDeg = 4
run_dp = True

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
unempl_rate = {
    2: 0.1431,
    3: 0.1132,
    4: 0.0703 
}

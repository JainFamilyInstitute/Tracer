import numpy as np
from constants import N_SIM, unemp_frac, education_level, start_ages, RETIRE_AGE, MU, END_AGE, unemp_rate, ret_frac, ISA_POVERTY_MULTIPLIER, IDR_POVERTY_MULTIPLIER, POVERTY_LEVEL, TERM_EXT, NOMINAL_CAP_MULTIPLIER
from file_handlers import read_ids, read_age_coeffs, read_variances
from scipy.stats import bernoulli


def init_income_process(alt_deg):
    coeff_this_group = read_age_coeffs(alt_deg)
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(start_ages[alt_deg], RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    return income

def base_income_process(*, alt_deg):
    # generate random walk and normal r.v.
    # NB: volatility reduction in sigma_perm, sigma_tran
    income = init_income_process(alt_deg)
    sigma_perm, sigma_tran = read_variances(alt_deg)
    
    np.random.seed(0)
    rn_perm = np.random.normal(MU, 0.75*sigma_perm, (N_SIM, RETIRE_AGE - start_ages[alt_deg] + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    np.random.seed(1)
    rn_tran = np.random.normal(MU, 0.75*sigma_tran, (N_SIM, RETIRE_AGE - start_ages[alt_deg] + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income)

    # - retirement
    ret_income_vec = ret_frac[alt_deg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # unemployment risk
    # generate bernoulli random variable
    np.random.seed(seed=2)
    p = 1 - unemp_rate[alt_deg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - start_ages[alt_deg] + 1, N_SIM)).astype(float)
    r[r == 0] = unemp_frac[alt_deg]
    ones = np.ones((END_AGE - RETIRE_AGE, N_SIM))
    bern = np.append(r, ones, axis=0)
    Y = np.multiply(inc_with_inc_risk, bern.T)
    return Y

def income_adjustment_trivial(*, Y, **kwargs):
    return Y

def income_adjustment_isa(*, Y, alt_deg, term, principal, share_pct, **kwargs):
    # adjust income with ISA
    income_threshold = ISA_POVERTY_MULTIPLIER * POVERTY_LEVEL
    term_ub = term + TERM_EXT

    nominal_cap = np.ones(N_SIM) * NOMINAL_CAP_MULTIPLIER * principal
    P = np.zeros_like(Y)
    P_cumsum = np.zeros_like(Y)

    # t = 0
    cond_zero_pmt = Y[:, 0] < income_threshold
    P[cond_zero_pmt, 0] = 0
    cond_nonzero_pmt = np.logical_not(cond_zero_pmt)
    P[cond_nonzero_pmt, 0] = np.minimum(Y[cond_nonzero_pmt, 0] * share_pct, nominal_cap[cond_nonzero_pmt])
    P_cumsum[:, 0] = P[:, 0]

    # P[:, 0] = np.where(cond_zero_pmt, 0, )

    for t in range(1, END_AGE - start_ages[alt_deg] + 1):

        cond1 = Y[:, t] < income_threshold
        cond2 = np.count_nonzero(P[:, :t], axis=1) >= term      # count non-zero from 0 to t-1
        cond3 = np.array([True if t >= term_ub else False] * N_SIM)
        cond_zero_pmt = np.logical_or(np.logical_or(cond1, cond2), cond3)
        P[cond_zero_pmt, t] = 0

        cond_nonzero_pmt = np.logical_not(cond_zero_pmt)
        P[cond_nonzero_pmt, t] = np.minimum(Y[cond_nonzero_pmt, t] * share_pct,
                                            nominal_cap[cond_nonzero_pmt] - P_cumsum[cond_nonzero_pmt, t-1])

        P_cumsum[:, t] = P_cumsum[:, t-1] + P[:, t]

    adj_Y = Y - P
    return adj_Y
    # return adj_Y, P, Y

def income_adjustment_loan(*, Y, alt_deg, principal, payment, **kwargs):
     # adjust income with debt repayment
    D = np.zeros(Y.shape)
    D[:, 0] = principal
    P = np.zeros(Y.shape)

    for t in range(END_AGE - start_ages[alt_deg]):
        cond1 = np.logical_and(Y[:, t] >= 2 * payment, D[:, t] >= payment)
        cond2 = np.logical_and(Y[:, t] >= 2 * D[:, t], D[:, t] < payment)
        cond3 = np.logical_and(Y[:, t] < 2 * payment, D[:, t] >= payment)
        cond4 = np.logical_and(Y[:, t] < 2 * D[:, t], D[:, t] < payment)

        P[cond1, t] = payment
        P[cond2, t] = D[cond2, t] * (1 + rate)
        P[cond3, t] = Y[cond3, t] / 2
        P[cond4, t] = Y[cond4, t] / 2

        D[:, t + 1] = D[:, t] * (1 + rate) - P[:, t]
        D[cond2, t + 1] = 0
    adj_Y = Y - P
    return adj_Y

def income_adjustment_idr(*, Y, alt_deg, principal, payment, **kwargs):
    # def adj_income_process(income, sigma_perm, sigma_tran, INIT_DEBT, N_SIM, path=None):
    # adjust income with debt repayment
    D = np.zeros(Y.shape)
    D[:, 0] = principal

    idr_cutoff = IDR_POVERTY_MULTIPLIER * POVERTY_LEVEL

    P = (Y - idr_cutoff)
    P = np.where(P < 0, 0, P)
    P = P * 0.15   # income share percentage !!!
    P[:, 20:] = 0

    for t in range(END_AGE - start_ages[alt_deg]):
        if t < 20:
            cond0 = P[:, t] >= D[:, t]
            P[cond0, t] = D[cond0, t]
            # D[cond0, t] = 0
            cond1 = np.logical_and(P[:, t] >= D[:, t] * rate, P[:, t] < D[:, t])
            D[cond1, t + 1] = D[cond1, t] * (1 + rate) - P[cond1, t]
            cond2 = P[:, t] < D[:, t] * rate
            D[cond2, t + 1] = D[cond2, t] + (D[cond2, t] * rate - P[cond2, t]) / 2
        else:
            D[:, t] = 0


    adj_Y = Y - P
    return adj_Y
    # return adj_Y, P, Y, D
    # return Y, P, D

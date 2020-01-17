from constants import unemp_frac, education_level, start_ages, RETIRE_AGE, MU, END_AGE, unemp_rate, N_C, ret_frac
from scipy.stats import bernoulli

def cal_income(coeffs, alt_deg):
    coeff_this_group = coeffs.loc[education_level[alt_deg]]
    a  = coeff_this_group['a']
    b1 = coeff_this_group['b1']
    b2 = coeff_this_group['b2']
    b3 = coeff_this_group['b3']

    ages = np.arange(start_ages[alt_deg], RETIRE_AGE+1)      # 22 to 65

    income = (a + b1 * ages + b2 * ages**2 + b3 * ages**3)  # 0:43, 22:65
    return income

def base_income_process(income, sigma_perm, sigma_tran, n_sim, alt_deg):
    # generate random walk and normal r.v.
    np.random.seed(0)
    rn_perm = np.random.normal(MU, sigma_perm, (n_sim, RETIRE_AGE - start_ages[alt_deg] + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    np.random.seed(1)
    rn_tran = np.random.normal(MU, sigma_tran, (n_sim, RETIRE_AGE - start_ages[alt_deg] + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income)

    # - retirement
    ret_income_vec = ret_frac[alt_deg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # unemployment risk
    # generate bernoulli random variable
    np.random.seed(seed=2)
    p = 1 - unemp_rate[alt_deg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - start_ages[alt_deg] + 1, n_sim)).astype(float)
    r[r == 0] = unemp_frac[alt_deg]
    ones = np.ones((END_AGE - RETIRE_AGE, n_sim))
    bern = np.append(r, ones, axis=0)
    Y = np.multiply(inc_with_inc_risk, bern.T)
    return Y

def adj_income_process_isa(Y, term, rho):
    # adjust income with ISA
    Y[:, :term] *= rho
    return Y


def adj_income_process_loan(Y, alt_deg, principal, payment):
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
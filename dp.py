import numpy as np
import pandas as pd
import time
from utilities import utility, exp_val_new
from constants import start_ages, END_AGE, LOWER_BOUND_W, UPPER_BOUND_W, gamma_exp_frac, N_W, N_C, R, DELTA


def dp_solver(*, adj_income, cond_prob, gamma, n_sim, alt_deg):
    ###########################################################################
    #                                Setup                                    #
    ###########################################################################
    START_AGE = start_ages[alt_deg]
    # construct grids
    even_grid = np.linspace(0, 1, N_W)
    grid_w = LOWER_BOUND_W + (UPPER_BOUND_W - LOWER_BOUND_W) * even_grid**gamma_exp_frac[gamma]

    # initialize arrays for value function and consumption
    v = np.zeros((2, N_W))
    c = np.zeros((2, N_W))

    # terminal period: consume all the wealth
    ut = utility(grid_w, gamma)
    v[0, :] = ut
    c[0, :] = grid_w

    # collect results
    col_names = [str(age + START_AGE) for age in range(END_AGE-START_AGE, -1, -1)]     # 100 to 22
    c_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    v_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    c_collection[str(END_AGE)] = c[0, :]
    v_collection[str(END_AGE)] = v[0, :]

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')
        start_time = time.time()
        for i in range(N_W):
            # print('wealth_grid_progress: ', i / N_W * 100)
            consmp = np.linspace(0.1, grid_w[i], N_C)
            u_r = utility(consmp, gamma)
            u_r = u_r[None].T

            savings = grid_w[i] - np.linspace(0, grid_w[i], N_C)
            savings_incr = savings * (1 + R)
            savings_incr = savings_incr[None].T

            expected_value = exp_val_new(adj_income[:, t], savings_incr, grid_w, v[0, :], n_sim)

            v_array = u_r + DELTA * cond_prob[t] * expected_value    # v_array has size N_C-by-1
            v[1, i] = np.max(v_array)
            pos = np.argmax(v_array)
            c[1, i] = consmp[pos]

        # dump consumption array and value function array
        c_collection[str(t + START_AGE)] = c[1, :]
        v_collection[str(t + START_AGE)] = v[1, :]

        # change v & c for calculation next stage
        v[0, :] = v[1, :]

        print("--- %s seconds ---" % (time.time() - start_time))

    return c_collection, v_collection


from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from constants import N_C, N_SIM

###########################################################################
#                              Functions                                  #
###########################################################################


def utility(values, gamma):
    if gamma == 1:
        return np.log(values)
    else:
        return values**(1-gamma) / (1-gamma)


def exp_val_new(y, savings_incr, grid_w, v):

    COH = np.zeros((N_SIM, N_C))
    COH[:] = np.squeeze(savings_incr)
    COH += y[None].T

    COH[COH > grid_w[-1]] = grid_w[-1]
    COH[COH < grid_w[0]] = grid_w[0]

    spline = CubicSpline(grid_w, v, bc_type='natural')  # minimum curvature in both ends

    # p = mp.Pool(processes=mp.cpu_count())
    # v_w = p.apply(spline, args=(COH,))
    # p.close()

    v_w = np.zeros((N_SIM, N_C))
    for i in range(N_SIM):
        v_w[i, :] = spline(COH[i, :])

    ev = v_w.mean(axis=0)
    return ev[None].T


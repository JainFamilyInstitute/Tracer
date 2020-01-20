import pandas as pd, os
from constants import income_fn, surviv_fn, isa_loop_fn, debt_loop_fn, id_fn, ce_fn, education_level


# isa_loop_fp = os.path.join(base_path, 'data', isa_loop_fn)
# debt_loop_fp = os.path.join(base_path, 'data', debt_loop_fn)
# ce_fp = os.path.join(base_path, 'results', ce_fn)

def read_ids_for_alt_deg(alt_deg):
    # ids for specified alt_deg
    base_path = os.path.dirname(__file__)
    id_fp = os.path.join(base_path, 'data', id_fn)
    ids = pd.read_excel(id_fp)
    ids = ids[ids['AltDeg'] == education_level[alt_deg]]
    return ids['ID']

def read_age_coeffs(alt_deg):
    # age coefficients
    base_path = os.path.dirname(__file__)
    income_fp = os.path.join(base_path, 'data', income_fn)
    coeff_df = pd.read_excel(income_fp, sheet_name='Coefficients', index_col=0)
    return coeff_df.loc[education_level[alt_deg]]

def read_variance(alt_deg):
    # decomposed variance
    base_path = os.path.dirname(__file__)
    income_fp = os.path.join(base_path, 'data', income_fn)
    std = pd.read_excel(income_fp, sheet_name='Variance', header=[1, 2]) 
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])
    return std['Labor Income Only'][education_level[alt_deg]]

def read_survival():
    # conditional survival probabilities
    base_path = os.path.dirname(__file__)
    mortal_fp = os.path.join(base_path, 'data', surviv_fn)
    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)
    return cond_prob['CSP']
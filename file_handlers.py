import pandas as pd, os
from constants import coeff_fn, surviv_fn, isa_loop_fn, debt_loop_fn, id_fn, ce_fn, education_level, income_output_fn


# isa_loop_fp = os.path.join(base_path, 'data', isa_loop_fn)
# debt_loop_fp = os.path.join(base_path, 'data', debt_loop_fn)
# ce_fp = os.path.join(base_path, 'results', ce_fn)

def make_path(fn, dir = 'data'):
    base_path = os.path.dirname(__file__)
    fp = os.path.join(base_path, dir, fn)
    return fp

def read_ids(alt_deg):
    # ids for specified alt_deg
    ids = pd.read_excel(make_path(id_fn))
    ids = ids[ids['AltDeg'] == education_level[alt_deg]]
    return ids['ID']

def read_age_coeffs(alt_deg):
    # age coefficients
    coeff_df = pd.read_excel(make_path(coeff_fn), sheet_name='Coefficients', index_col=0)
    return coeff_df.loc[education_level[alt_deg]]

def read_variances(alt_deg):
    # decomposed variance
    std = pd.read_excel(make_path(coeff_fn), sheet_name='Variance', header=[1, 2]) 
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])
    return std['Labor Income Only'][education_level[alt_deg]]

def read_survival():
    # conditional survival probabilities
    cond_prob = pd.read_excel(make_path(surviv_fn))
    cond_prob.set_index('AGE', inplace=True)
    return cond_prob['CSP']

def export_base_incomes(base_incomes, alt_deg, method=None):
    # dump intermediate income profiles to csv
    output_fn = income_output_fn+str(alt_deg) + ('' if method == None else str(method) + '.csv')
    pd.DataFrame(data=base_incomes).to_csv(make_path(output_fn, dir='incomes'))
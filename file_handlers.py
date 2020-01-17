import pandas as pd

def read_ids_for_alt_deg(id_fp, alt_deg):
    # ids for specified alt_deg
    ids = pd.read_excel(id_fp)
    ids = ids[ids['AltDeg'] == education_level[alt_deg]]
    return ids

def read_age_coeff(income_fp):
    # age coefficients
    age_coeff = pd.read_excel(income_fp, sheet_name='Coefficients', index_col=0)
    return age_coeff

def read_variance(income_fp):
    # decomposed variance
    std = pd.read_excel(income_fp, sheet_name='Variance', header=[1, 2], index_col=0)
    std.reset_index(inplace=True)
    std.drop(std.columns[0], axis=1, inplace=True)
    std.drop([1, 3], inplace=True)
    std.index = pd.CategoricalIndex(['sigma_permanent', 'sigma_transitory'])
    return std

def read_survival(mortal_fp):
    # conditional survival probabilities
    cond_prob = pd.read_excel(mortal_fp)
    cond_prob.set_index('AGE', inplace=True)
    return cond_prob
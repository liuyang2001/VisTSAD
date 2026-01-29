import numpy as np
import pandas as pd

ROLLING_VAR_WINDOW = 16

def compute_eu_matrix(data):
    # data shape: (N_vars, Window_Size)
    diff = data[:, None, :] - data[None, :, :]
    return np.sqrt(np.mean(diff**2, axis=2))

def mutual_info_binning(x, y, bins=16):
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / (np.sum(c_xy) + 1e-12)
    p_x, p_y = np.sum(p_xy, axis=1), np.sum(p_xy, axis=0)
    mask = (p_xy > 0) & (p_x[:, None] * p_y[None, :] > 0)
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / (p_x[:, None] * p_y[None, :])[mask]))
    return mi

def compute_mi_matrix(data):
    D, L = data.shape
    matrix = np.zeros((D, D))
    for i in range(D):
        for j in range(i, D):
            val = mutual_info_binning(data[i], data[j])
            matrix[i, j] = matrix[j, i] = val
    return matrix

def get_6_matrices(window_data_norm):

    mv = window_data_norm
    n_vars = mv.shape[0] 
    
    md = np.zeros_like(mv)
    md[:, 1:] = mv[:, 1:] - mv[:, :-1]
    
    mvar = np.zeros_like(mv)
    for i in range(n_vars): 
        mvar[i] = pd.Series(mv[i]).rolling(ROLLING_VAR_WINDOW, min_periods=1).var().fillna(0).values
    
    return {
        "eu_val": compute_eu_matrix(mv), 
        "eu_diff": compute_eu_matrix(md), 
        "eu_var": compute_eu_matrix(mvar),
        "mi_val": compute_mi_matrix(mv), 
        "mi_diff": compute_mi_matrix(md), 
        "mi_var": compute_mi_matrix(mvar)
    }
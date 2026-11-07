import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_weight_matching

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Safe distance matrix
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    # Base heuristic: 1 / d^p
    p = 2.0
    eta_base = 1.0 / (d_safe ** p)
    
    # Compute Minimum Weight Perfect Matching
    # Weights = -d^2
    weights = -d_safe ** 2
    np.fill_diagonal(weights, 0)
    
    A = csr_matrix(weights)
    matching = maximum_weight_matching(A)
    
    # Apply boost
    boost = 100.0
    eta = eta_base.copy()
    for i, j in matching:
        eta[i, j] *= boost
        eta[j, i] *= boost
        
    # Finalize
    np.fill_diagonal(eta, 1e-9)
    eta[eta <= 0] = 1e-9
    
    return eta

import numpy as np

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # Remaining space if item i is the last in the bin
    # R[i] = capacity - demands[i]
    R = capacity - demands
    
    # Pair sums: demands[i] + demands[j]
    # We need to check if they fit
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid mask: pair_sums <= capacity
    valid = pair_sums <= capacity
    
    # Heuristic: d_j / R_i
    # demands[None, :] is d_j (broadcast over rows)
    # R[:, None] is R_i (broadcast over cols)
    # Note: R is 1D array of shape (n,).
    # We want eta[i, j] = demands[j] / R[i]
    
    # Avoid division by zero (though R[i] >= 150-99=51 > 0)
    eta = np.zeros((n, n), dtype=float)
    
    # Calculate for valid entries
    # We can do this element-wise
    # eta[i, j] = demands[j] / R[i]
    
    # Using broadcasting
    # d_j matrix: demands[None, :] shape (1, n) -> broadcast to (n, n)
    # R_i matrix: R[:, None] shape (n, 1) -> broadcast to (n, n)
    
    # Create full matrices for calculation
    d_j_mat = demands[None, :]
    R_i_mat = R[:, None]
    
    # Calculate ratio
    ratio = d_j_mat / R_i_mat
    
    # Apply mask
    eta = np.where(valid, ratio, 1e-9)
    
    return eta

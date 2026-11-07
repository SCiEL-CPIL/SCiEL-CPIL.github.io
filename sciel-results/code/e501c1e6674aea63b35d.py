import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    D = distance_matrix
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    
    # Calculate savings S_ij = D_0i + D_0j - D_ij
    # D_0 is the first row
    D0 = D[0, :]
    
    # Create matrices for broadcasting
    # D0_i is D0[i], D0_j is D0[j]
    # We can use broadcasting
    D0_i = D0[:, np.newaxis]
    D0_j = D0[np.newaxis, :]
    
    # S_ij
    S = D0_i + D0_j - D
    
    # Denominator for normalization
    Denom = D0_i + D0_j
    
    # Avoid division by zero in Denom (should be > 0 for i,j != 0)
    # For i=0 or j=0, S=0, so S_norm=0.
    # If Denom is 0 (only if D0i=0 and D0j=0, i.e., i=j=0 or depot at same loc), handle it.
    Denom_safe = np.where(Denom > eps, Denom, 1.0)
    
    S_norm = S / Denom_safe
    
    # Base heuristic: Inverse squared distance
    # D_safe to avoid 0
    D_safe = np.where(D > eps, D, 1.0)
    base = 1.0 / (D_safe ** 2)
    
    # Demand weight
    # d_j is the demand of the destination node j
    # demands is a 1D array
    d_j = demands[np.newaxis, :] # Shape (1, N)
    demand_weight = 1.0 + 0.1 * d_j
    
    # Savings weight
    savings_weight = 1.0 + 2.0 * S_norm
    
    # Combine
    eta = base * demand_weight * savings_weight
    
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    # Replace non-finite
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta

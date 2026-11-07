"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Validity mask: item j can follow item i if their sum <= capacity
    valid = pair_sum <= capacity
    
    # Calculate waste for valid pairs
    waste = capacity - pair_sum
    waste = np.where(valid, waste, 0)
    
    # Fit heuristic: prefer less waste with sharper discrimination.
    # Using 1/(1+waste)^2 gives stronger penalty for larger waste,
    # making tight fits (small waste) much more attractive than loose fits.
    fit = 1.0 / (1.0 + waste) ** 2
    
    # Scarcity heuristic: prefer items that have fewer compatible partners
    # compat_count[j] is the number of items i such that d_i + d_j <= capacity
    compat_count = valid.sum(axis=0).astype(float)
    
    # Use inverse square for stronger scarcity signal
    scarcity = 1.0 / (compat_count ** 2)
    
    # Combine heuristics
    # eta[i, j] is attractiveness of placing j after i
    # We multiply fit (depends on i, j) by scarcity (depends on j)
    eta = fit * scarcity[None, :]
    
    # Set invalid entries to 1e-9
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Simple deadline guard (though computation is fast)
    if time.time() - start_time > 250:
         # Fallback to seed-like heuristic if somehow slow (unlikely)
         pair = demands[:, None] + demands[None, :]
         return np.where(pair <= capacity, pair / capacity, 1e-9)
         
    return eta
# EVOLVE-BLOCK-END

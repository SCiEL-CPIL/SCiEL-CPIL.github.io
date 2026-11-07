import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Feasibility mask
    feasible = pair_sum <= capacity
    
    # Calculate heuristic
    # We want to minimize waste: waste = capacity - pair_sum
    # Heuristic = 1 / (waste + 1)
    # For infeasible, set to 1e-9
    
    # Avoid division by zero or negative numbers
    waste = capacity - pair_sum
    # Only compute where feasible
    # For infeasible, waste is negative, so 1/(waste+1) might be weird.
    # We explicitly mask.
    
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Compute for feasible pairs
    # Use a steeper penalty for waste to strongly prefer tight packing
    # H = 1 / (waste^alpha) where alpha > 1 amplifies preference for small gaps
    # But need to handle waste=0 case (perfect fit)
    
    # For feasible pairs: waste is in [0, capacity]
    # Use H = 1 / (waste + epsilon) with small epsilon to avoid div by 0
    # This creates a much stronger preference for near-perfect fits
    eps = 1e-6
    valid_waste = waste[feasible] + eps
    H[feasible] = 1.0 / valid_waste
    
    # Scale to keep values in reasonable range
    # The max value will be 1/eps = 1e6, which might be too large
    # Normalize by dividing by max
    max_h = np.max(H)
    if max_h > 0 and np.isfinite(max_h):
        H = H / max_h
        # Add small base to ensure all positive
        H = H + 1e-3
    
    # Check for non-finite or non-positive
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Time guard (unlikely to be needed)
    if time.time() - start_time > 300:
         # Return a safe fallback if somehow slow (shouldn't happen)
         return np.ones((n, n)) * 1e-9

    return H

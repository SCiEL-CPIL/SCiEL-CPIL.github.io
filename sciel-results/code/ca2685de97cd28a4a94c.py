import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate waste
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix
    H = np.zeros((n, n), dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if np.any(fits):
        # Use a smoother heuristic that balances tight packing with feasibility
        # The idea: reward pairs that leave moderate waste (not too much, not zero)
        # Use a Gaussian-like kernel centered at optimal waste level
        
        # For capacity=150, items 20-99, pair sums range from 40 to 198
        # Fitting pairs have sum <= 150, so waste in [0, 110]
        # Ideal waste is small but not zero (to allow flexibility)
        
        eps = 1e-6
        valid_waste = waste[fits] + eps
        
        # Try different heuristic shapes:
        # 1) Inverse waste (strong preference for tight fits)
        # 2) Inverse waste squared (even stronger)
        # 3) Exponential decay
        # 4) A combination that rewards moderate waste
        
        # Use a smooth function that peaks at small waste but doesn't blow up
        # h(w) = exp(-w / sigma) where sigma controls the width
        # This is smooth and bounded, avoids numerical issues
        
        # Try sigma values: smaller sigma means stronger preference for tight fits
        # Let's use a moderate sigma to balance exploration and exploitation
        
        # Based on the problem scale (waste 0-110), sigma around 20-30 seems reasonable
        sigma = 25.0
        
        # Exponential decay heuristic
        H[fits] = np.exp(-valid_waste / sigma)
        
        # Ensure all values are finite and positive
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
    
    return H

import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    n = len(demands)
    d = np.array(demands, dtype=float)
    c = float(capacity)
    
    # Sum of pairs
    pair_sums = d[:, None] + d[None, :]
    
    # Feasibility mask
    feasible = pair_sums <= c
    
    # Residual space
    residuals = c - pair_sums
    
    # Smallest item size
    w_min = np.min(d)
    
    # Avoid division by zero if w_min is 0 (though demands are 20-99, so w_min >= 20)
    if w_min == 0:
         w_min = 1.0
         
    # Smooth structural fit: reward residuals close to integer multiples of w_min
    # Use a Gaussian kernel: exp(-0.5 * (residual - k*w_min)^2 / sigma^2) for nearest k
    # This is smooth and differentiable, unlike modulo
    
    # Distance to nearest multiple of w_min
    k = np.round(residuals / w_min)
    distance_to_multiple = np.abs(residuals - k * w_min)
    
    # Gaussian kernel with width proportional to w_min
    sigma = w_min * 0.15
    structural_fit = np.exp(-0.5 * (distance_to_multiple / sigma) ** 2)
    
    # Tightness score: prefer larger sums (fills bin more)
    tightness = pair_sums / c
    
    # Combined heuristic: balance tightness and structural fit
    # Weight: 60% tightness, 40% structural fit
    h = 0.6 * tightness + 0.4 * structural_fit
    
    # Apply feasibility
    h = np.where(feasible, h, 1e-9)
    
    # Ensure no negative values and handle non-finite values
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        # Return best available result
        return h
    
    return h

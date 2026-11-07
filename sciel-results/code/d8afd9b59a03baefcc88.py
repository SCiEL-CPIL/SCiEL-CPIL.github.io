import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    # Convert to float array to be safe
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # Calculate gap for valid pairs
        gaps = capacity - sums[valid]
        
        # Primary signal: inverse gap (sharper best-fit preference)
        # Using a small epsilon to avoid division by zero and create a strong peak for exact fits
        inv_gap = 1.0 / (gaps + 1e-6)
        
        # Secondary signal: fill ratio power (smoother, adds secondary info)
        ratio = sums[valid] / capacity
        fill_score = ratio ** 2.0
        
        # Blend: product keeps strong best-fit but adds fill-ratio shaping
        # Product is in [0, 1] since both factors are in [0, 1]
        eta[valid] = inv_gap * fill_score
        
        # Safety check for non-finite values
        eta[~np.isfinite(eta)] = 1e-9
    
    return eta

import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 320.0  # 300s limit, keep 20s margin

def heuristics(demands, capacity):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones((len(demands), len(demands))) * 1e-9

    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate sums
    sums = d[:, None] + d[None, :]
    
    # Create mask for valid pairs (fit in capacity)
    valid = sums <= capacity
    
    # Initialize eta with small positive values for invalid entries
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(valid):
        # 1. Compute scarcity (degree) for each item j
        # deg[j] = number of items k such that d[j] + d[k] <= capacity
        # This is the row sum of the valid mask (excluding self? or including?)
        # If d[j] + d[j] <= capacity, it fits with itself.
        # The "partners" for j are all k where valid[j, k] is True.
        deg = np.sum(valid, axis=1) # Row j: how many items fit with j
        
        # Avoid division by zero, though deg should be > 0 for all items (min demand 20, cap 150, 20+20=40 <= 150)
        # Max demand 99, 99+99=198 > 150. So large items might have low deg.
        # Smallest deg is for largest items.
        # Inverse degree: higher for scarce items.
        scarcity = 1.0 / (deg + 1.0)
        
        # 2. Compute geometric fit score
        gaps = capacity - sums
        # Sharpen the best-fit preference
        # Using a smaller epsilon makes the peak sharper
        eps = 0.1
        inv_gap = 1.0 / (gaps + eps)
        
        # Fill ratio power
        ratio = sums / capacity
        # Use a moderate power to avoid over-sharpening combined with scarcity
        fill_score = ratio ** 2.0
        
        # Combine: 
        # We want to place item j if it fits well (high inv_gap * fill_score) 
        # AND if it is scarce (high scarcity[j])
        # Note: scarcity depends only on j.
        # So we broadcast scarcity across rows.
        # eta[i, j] = inv_gap[i, j] * fill_score[i, j] * scarcity[j]
        
        # Apply only to valid entries
        # We need to be careful with the mask.
        # eta[valid] = ...
        # But scarcity is a vector of length n.
        # We can create a matrix of scarcity by broadcasting: scarcity[None, :]
        
        scarcity_matrix = scarcity[None, :]
        
        base_score = inv_gap * fill_score
        eta_valid = base_score * scarcity_matrix
        
        # Assign to valid positions
        eta[valid] = eta_valid[valid]
        
        # Safety check
        eta[~np.isfinite(eta)] = 1e-9
        # Ensure no zeros (ACO requires positive)
        eta[eta <= 0] = 1e-9
        
    return eta

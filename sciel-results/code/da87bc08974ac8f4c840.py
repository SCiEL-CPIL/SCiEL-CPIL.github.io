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
    
    # Fit heuristic: Prefer less waste.
    # Sharper penalty: Cubic? Let's stick to Square first, it's robust.
    fit = 1.0 / (1.0 + waste) ** 2
    
    # Scarcity heuristic: Prefer items that have fewer compatible partners.
    # degree[j] = number of items i such that d_i + d_j <= capacity
    degree = valid.sum(axis=0).astype(float)
    # Avoid division by zero, though degree should be >= 1 for valid items
    # If degree is 0, the item is isolated (too big for anything including itself? 
    # But d_j <= capacity is guaranteed? "demands are integers from 20 to 99", capacity 150.
    # So d_j < capacity. So at least j fits with itself? 
    # Wait, valid[i,j] is d_i + d_j <= capacity.
    # If d_j > capacity/2, it can only pair with small items.
    # If d_j <= capacity/2, it can pair with many.
    # Degree will be at least 1 (pairing with smallest item? No, pairing with itself? 
    # If 2*d_j <= capacity, then valid[j,j] is True.
    # If 2*d_j > capacity, it might still pair with smaller items.
    # Is it possible for degree to be 0? 
    # Min demand 20, Max 99. Cap 150.
    # 99+99 = 198 > 150.
    # 99+20 = 119 <= 150.
    # So even the largest item can pair with the smallest.
    # So degree >= 1 for all items.
    
    # Use linear inverse for scarcity to avoid extreme values
    scarcity = 1.0 / (degree + 1.0)
    
    # Size heuristic: Prefer larger items (more constrained)
    # Using raw demand or normalized? 
    # Let's use raw demand to emphasize size difference.
    size_factor = demands
    
    # Combine: 
    # eta[i, j] = fit[i, j] * scarcity[j] * size_factor[j]
    eta = fit * scarcity[None, :] * size_factor[None, :]
    
    # Set invalid entries to 1e-9
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Deadline guard
    if time.time() - start_time > 250:
         pair = demands[:, None] + demands[None, :]
         return np.where(pair <= capacity, pair / capacity, 1e-9)
         
    return eta

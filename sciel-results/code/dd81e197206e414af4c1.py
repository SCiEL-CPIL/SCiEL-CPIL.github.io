import numpy as np
import time

def heuristics(prizes, weights):
    # Start time to respect the deadline guard requirement, although computation is fast
    start_time = time.time()
    deadline = 300.0 - 5.0 # Safety margin
    
    try:
        # prizes: (5, 100), weights: (5, 100)
        
        # 1. Calculate Total Prize per item
        total_prize = prizes.sum(axis=0) # shape (100,)
        
        # Normalize total prize to [0, 1]
        tp_min = np.min(total_prize)
        tp_max = np.max(total_prize)
        tp_range = tp_max - tp_min
        if tp_range < 1e-9:
            tp_range = 1e-9
        norm_prize = (total_prize - tp_min) / tp_range # shape (100,)
        
        # 2. Normalize Weights per constraint to [0, 1]
        # w_min, w_max shape (5, 1)
        w_min = np.min(weights, axis=1, keepdims=True)
        w_max = np.max(weights, axis=1, keepdims=True)
        w_range = w_max - w_min
        # Avoid division by zero
        w_range = np.where(w_range < 1e-9, 1e-9, w_range)
        
        norm_weights = (weights - w_min) / w_range # shape (5, 100)
        
        # 3. Find Maximum Normalized Weight per item across constraints
        # This represents the "heaviest" relative resource consumption for the item
        max_norm_weight = np.max(norm_weights, axis=0) # shape (100,)
        
        # 4. Compute Heuristic: norm_prize * (1 / max_norm_weight)
        # Items with high total prize and low worst-case relative weight are most attractive
        max_norm_weight_safe = np.maximum(max_norm_weight, 1e-9)
        heuristic = norm_prize / max_norm_weight_safe
        
        # Ensure finite and positive
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)
        
        # Check time (just in case, though it's fast)
        if time.time() > deadline:
             # Return a fallback if somehow it took too long (unlikely)
             return np.ones(100)
             
        return heuristic
        
    except Exception:
        # Fallback to a simple heuristic if something goes wrong
        return np.ones(100)

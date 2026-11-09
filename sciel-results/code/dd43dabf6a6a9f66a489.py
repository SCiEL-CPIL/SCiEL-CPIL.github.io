"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety, though not strictly needed for N=100
_start_time = time.time()

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Safety check: if we've been running too long, fall back to simple greedy
    # 150s limit. We are doing ~100 * 100 * 3 = 30,000 ops per call.
    # 100 calls = 3,000,000 ops. Should be < 0.1s.
    # But just in case, if n is unexpectedly large, we might need a guard.
    # For now, assume N is small as per task description (100 items).
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Pre-compute sorting orders for the 3 heuristics
    # 1. Density (Value / Weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    # Items with 0 weight should have infinite density. 
    # argsort handles inf.
    order_density = np.argsort(-densities)
    
    # 2. Value (Highest Value first)
    order_value = np.argsort(-values)
    
    # 3. Weight (Lightest first)
    order_weight = np.argsort(weights)
    
    best_val = -1.0
    best_idx = -1
    
    for i in range(n):
        # Check if item i is feasible (should be, but just in case)
        if weights[i] > remaining_capacity:
            continue
            
        cap = remaining_capacity - weights[i]
        base_val = values[i]
        
        max_sim_val = 0.0
        
        # Simulate 1: Density Greedy
        sim_cap = cap
        sim_val = 0.0
        for j in order_density:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
        if sim_val > max_sim_val:
            max_sim_val = sim_val
            
        # Simulate 2: Value Greedy
        sim_cap = cap
        sim_val = 0.0
        for j in order_value:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
        if sim_val > max_sim_val:
            max_sim_val = sim_val
            
        # Simulate 3: Weight Greedy
        sim_cap = cap
        sim_val = 0.0
        for j in order_weight:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
        if sim_val > max_sim_val:
            max_sim_val = sim_val
            
        total_val = base_val + max_sim_val
        
        if total_val > best_val:
            best_val = total_val
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END

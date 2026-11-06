import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_deadline = None
_TIME_BUDGET = 140.0  # seconds, with 10s safety margin before 150s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + _TIME_BUDGET
    return time.time() > _deadline - 2.0  # 2s safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time, _deadline
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + _TIME_BUDGET
    
    # Check deadline
    if _check_deadline():
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Convert to numpy arrays for efficiency
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    
    # Find feasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_indices]
    feasible_values = values[feasible_indices]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 0:
        return -1
    
    # Compute value densities
    densities = feasible_values / feasible_weights
    
    # Strategy: Score each candidate by a combination of:
    # 1. Its own value
    # 2. Estimated value of items that can fit after taking it
    # 3. A "tightness" bonus for items that fill capacity well
    
    # For speed, use a greedy fill estimate but with a smarter ordering
    # Sort all feasible items by density descending for the fill estimation
    sort_order = np.argsort(densities)[::-1]
    sorted_weights = feasible_weights[sort_order]
    sorted_values = feasible_values[sort_order]
    
    # Precompute prefix sums for quick range queries
    prefix_weights = np.zeros(n_feasible + 1)
    prefix_values = np.zeros(n_feasible + 1)
    for i in range(n_feasible):
        prefix_weights[i+1] = prefix_weights[i] + sorted_weights[i]
        prefix_values[i+1] = prefix_values[i] + sorted_values[i]
    
    best_score = -1
    best_index = -1
    
    for i in range(n_feasible):
        wi = feasible_weights[i]
        vi = feasible_values[i]
        rem_cap = cap - wi
        
        if rem_cap < 0:
            continue
        
        # Estimate fill value: greedily add items by density that fit in rem_cap
        # But exclude item i itself
        # For speed, do a quick greedy pass
        
        fill_val = 0.0
        current_cap = rem_cap
        
        # Iterate through density-sorted items, skip item i
        for j in range(n_feasible):
            if j == i:
                continue
            wj = sorted_weights[j]
            vj = sorted_values[j]
            if wj <= current_cap:
                current_cap -= wj
                fill_val += vj
            else:
                # Don't break - smaller items might still fit
                # But for speed, we can break if we've gone through many
                pass
        
        total_val = vi + fill_val
        
        # Add a small bonus for items that use capacity efficiently
        # (higher density items that also have good packing)
        density_bonus = densities[i] * 0.1
        
        score = total_val + density_bonus
        
        if score > best_score:
            best_score = score
            best_index = i
    
    if best_index == -1:
        return -1
    
    return int(feasible_indices[best_index])

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item using Lagrangian relaxation to guide selection.
    
    Strategy:
    1. Find lambda via binary search such that the fractional knapsack
       solution (items with v_i/lambda > w_i) has total weight close to capacity.
    2. Use this lambda to rank items by v_i - lambda * w_i (Lagrangian score).
    3. For each top candidate, simulate greedy completion using the Lagrangian
       ranking and pick the best.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.array(remaining_weights, dtype=np.float64)
    values = np.array(remaining_values, dtype=np.float64)

    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = weights[feasible_mask]
    fv = values[feasible_mask]

    # Binary search for lambda that matches remaining capacity
    # In fractional knapsack, we fill items with v/w > lambda until capacity is met
    # Total weight of items with v_i > lambda * w_i should approximate remaining_capacity
    
    # Find range for lambda
    min_density = np.min(fv / fw)
    max_density = np.max(fv / fw)
    
    # Binary search for lambda
    lambda_lo = 0.0
    lambda_hi = max_density * 1.5
    
    def total_weight_at_lambda(lam):
        """Total weight of items that would be fully taken in fractional solution at lambda lam."""
        # Items with v_i > lam * w_i are fully taken
        mask = fv > lam * fw
        return np.sum(fw[mask])
    
    # Binary search: find lambda where total_weight_at_lambda ≈ remaining_capacity
    # As lambda increases, fewer items qualify, so total weight decreases
    for _ in range(40):  # 40 iterations for good precision
        lambda_mid = (lambda_lo + lambda_hi) / 2.0
        tw = total_weight_at_lambda(lambda_mid)
        if tw > remaining_capacity:
            lambda_lo = lambda_mid  # Need higher lambda to reduce weight
        else:
            lambda_hi = lambda_mid  # Can lower lambda to increase weight
    
    lambda_star = (lambda_lo + lambda_hi) / 2.0

    # Compute Lagrangian scores: v_i - lambda * w_i
    lag_scores = fv - lambda_star * fw
    
    # Sort by Lagrangian score descending
    sorted_lag_idx = np.argsort(-lag_scores)
    
    # Also compute density-based order as fallback
    densities = fv / fw
    sorted_density_idx = np.argsort(-densities)

    # Evaluate top candidates using greedy completion
    # Limit candidates to top 20 for efficiency
    max_candidates = min(20, len(feasible_indices))
    
    best_val = -1
    best_idx = -1
    
    for rank in range(max_candidates):
        k_local = sorted_lag_idx[rank]
        k = feasible_indices[k_local]
        
        # Simulate picking item k
        cap_rem = remaining_capacity - weights[k]
        val_acc = values[k]
        
        # Greedily fill using Lagrangian ordering
        for j in range(len(sorted_lag_idx)):
            idx_local = sorted_lag_idx[j]
            if idx_local == k_local:
                continue
            idx = feasible_indices[idx_local]
            if weights[idx] <= cap_rem:
                cap_rem -= weights[idx]
                val_acc += values[idx]
                if cap_rem <= 0:
                    break
        
        if val_acc > best_val:
            best_val = val_acc
            best_idx = k

    return int(best_idx)
# EVOLVE-BLOCK-END

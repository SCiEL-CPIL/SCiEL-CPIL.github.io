"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Lagrangian relaxation heuristic for 0/1 knapsack.
    
    We select the item maximizing v_i - lambda * w_i, where lambda is tuned
    so that the total weight of items with positive reduced cost (v_i > lambda * w_i)
    is approximately equal to the remaining capacity. This effectively solves
    a fractional knapsack relaxation to guide the 0/1 selection.
    
    If remaining_capacity is very small or no items are feasible, return -1.
    """
    # Handle edge cases: no items left
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter to only feasible items (weight <= remaining_capacity)
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_weights) == 1:
        return int(feasible_indices[0])
    
    # Compute densities
    densities = feasible_values / feasible_weights
    lambda_low = 0.0
    lambda_high = np.max(densities) + 1e-9
    
    # Check if all items fit
    total_weight_all = np.sum(feasible_weights)
    if total_weight_all <= remaining_capacity:
        # All items fit, pick the one with highest value
        best_local = np.argmax(feasible_values)
        return int(feasible_indices[best_local])
    
    # Binary search for lambda such that sum of weights of items with density > lambda
    # is close to remaining_capacity.
    
    def total_weight_for_lambda(lam):
        mask = densities > lam
        if not np.any(mask):
            return 0.0
        return np.sum(feasible_weights[mask])
    
    # Perform binary search on lambda
    for _ in range(60):  # 60 iterations for precision
        lambda_mid = (lambda_low + lambda_high) / 2.0
        tw = total_weight_for_lambda(lambda_mid)
        if tw > remaining_capacity:
            # Too many items, increase lambda
            lambda_low = lambda_mid
        else:
            # Too few items, decrease lambda
            lambda_high = lambda_mid
    
    lambda_opt = (lambda_low + lambda_high) / 2.0
    
    # Now, select the item with the highest reduced cost v_i - lambda_opt * w_i
    # among feasible items.
    reduced_costs = feasible_values - lambda_opt * feasible_weights
    
    # Pick the item with the highest reduced cost
    best_local = np.argmax(reduced_costs)
    
    # If the best reduced cost is negative, it might be better to stop
    if reduced_costs[best_local] <= 0:
        return -1
    
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END

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
    
    # Binary search for lambda such that sum of weights of items with v_i > lambda * w_i
    # is close to remaining_capacity.
    
    # The range of possible lambda values:
    # Lower bound: 0 (all items have positive reduced cost if lambda=0)
    # Upper bound: max(v_i / w_i) (no items have positive reduced cost if lambda > max density)
    
    densities = feasible_values / feasible_weights
    lambda_low = 0.0
    lambda_high = np.max(densities) + 1e-9
    
    # We want to find lambda such that the "knapsack fill" is close to remaining_capacity.
    # For a given lambda, the items selected are those with v_i - lambda * w_i > 0,
    # i.e., v_i / w_i > lambda.
    
    # Perform binary search on lambda
    # We want the total weight of items with density > lambda to be close to remaining_capacity.
    # As lambda increases, fewer items are selected, so total weight decreases.
    
    # Let's define a function that, given lambda, returns the total weight of items with density > lambda
    # and the sum of values.
    
    def total_weight_for_lambda(lam):
        mask = densities > lam
        if not np.any(mask):
            return 0.0
        return np.sum(feasible_weights[mask])
    
    # Check if at lambda=0, total weight <= remaining_capacity
    # If so, we can take all items, so lambda should be 0 (or very small)
    if total_weight_for_lambda(0.0) <= remaining_capacity:
        # All items fit, so we should pick the one with highest value (or any)
        # Actually, if all fit, we want to pick the one that gives most value, but since we're
        # picking one at a time, let's just pick the one with highest value.
        # But wait, the heuristic is supposed to guide selection. If all fit, we can just pick
        # the one with highest value.
        best_local = np.argmax(feasible_values)
        return int(feasible_indices[best_local])
    
    # Otherwise, we need to find a lambda such that the total weight is close to remaining_capacity.
    # Binary search
    for _ in range(50):  # 50 iterations for precision
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
    
    # If the best reduced cost is negative, it might be better to stop
    if np.max(reduced_costs) <= 0:
        return -1
    
    # Strategy: Among items with positive reduced cost, prefer those that fit well
    # in the remaining capacity. We'll score each item by:
    # - Primary: reduced cost (v_i - lambda * w_i)
    # - Secondary: how well it fits (items that exactly fill remaining space are good)
    
    # Find items with positive reduced cost
    positive_mask = reduced_costs > 0
    if not np.any(positive_mask):
        return -1
    
    pos_indices = np.where(positive_mask)[0]
    pos_reduced = reduced_costs[pos_indices]
    pos_weights = feasible_weights[pos_indices]
    pos_values = feasible_values[pos_indices]
    
    # Score: primarily by reduced cost, but boost items that can be combined with others
    # to fill the knapsack. A simple heuristic: items whose weight is close to 
    # remaining_capacity / k for small k are good, but let's use a different approach.
    
    # Better: among items with positive reduced cost, pick the one that maximizes
    # value density among those that fit, but weighted by how "critical" they are.
    
    # Actually, let's try: among positive reduced cost items, pick the one with 
    # highest value density (v/w) among those that fit, since high density items
    # are generally good to include early.
    
    # But we already filtered by reduced cost. Let's just pick the highest reduced cost.
    best_pos = np.argmax(pos_reduced)
    
    return int(feasible_indices[pos_indices[best_pos]])
# EVOLVE-BLOCK-END

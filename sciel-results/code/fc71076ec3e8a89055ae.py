"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: Use the fractional knapsack optimum as a surrogate for discrete choice.
    1. Sort items by value density (value/weight) in descending order.
    2. The optimal fractional solution fills items in this order until capacity is met.
    3. The "critical item" is the one that is partially filled (or the last fully filled one).
    4. We prefer items that are fully included in the fractional optimum.
    5. Among those, we pick the one with the highest value density (which is also the earliest in the sorted order).
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Compute value density
    with np.errstate(divide='ignore', invalid='ignore'):
        density = remaining_values / remaining_weights
    
    # Handle any zero or negative weights (shouldn't happen in valid instances, but be safe)
    # If weight is 0, density is infinite; if weight is negative, it's infeasible
    # For valid knapsack, weights > 0
    valid_mask = remaining_weights > 0
    if not np.any(valid_mask):
        return -1
    
    # Only consider valid items
    valid_indices = np.where(valid_mask)[0]
    valid_weights = remaining_weights[valid_mask]
    valid_values = remaining_values[valid_mask]
    valid_density = density[valid_mask]
    
    # Sort by density descending
    sort_order = np.argsort(-valid_density, kind='stable')
    
    # Find the critical item: walk through sorted items until capacity is exhausted
    # Items before the critical item are fully included in fractional optimum
    cum_weight = 0.0
    fully_included = set()
    critical_idx_in_sorted = -1
    
    for i in sort_order:
        w = valid_weights[i]
        if cum_weight + w <= remaining_capacity:
            cum_weight += w
            fully_included.add(i)
            critical_idx_in_sorted = i
        else:
            # This is the critical item (partially included)
            critical_idx_in_sorted = i
            break
    
    # The best item to pick is the highest density item that is fully included in the fractional optimum
    # This is the first item in the sorted order that is fully included
    # Since we sorted by density descending, the first item (index 0 in sort_order) has the highest density
    # We want the item with the highest density among fully_included
    
    # Since sort_order is sorted by density descending, we just need the first item in sort_order
    # that is in fully_included
    for idx_in_sorted in range(len(sort_order)):
        item_in_valid = sort_order[idx_in_sorted]
        if item_in_valid in fully_included:
            # Convert back to original index
            return int(valid_indices[item_in_valid])
    
    # If no item is fully included (all items exceed remaining capacity individually),
    # return -1 or pick the best feasible item
    # Actually, if an item doesn't fit, it's not feasible. The caller should only pass feasible items.
    # But to be safe, check if any item fits
    feasible_mask = valid_weights <= remaining_capacity
    if np.any(feasible_mask):
        # Pick the highest density feasible item
        feasible_in_valid = valid_indices[feasible_mask]
        feasible_density = valid_density[feasible_mask]
        best_local = np.argmax(feasible_density)
        return int(feasible_in_valid[best_local])
    
    return -1
# EVOLVE-BLOCK-END

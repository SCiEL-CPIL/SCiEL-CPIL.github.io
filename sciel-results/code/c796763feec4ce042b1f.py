"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: Use a modified value-density heuristic that accounts for
    capacity constraints. We prioritize items with high value density,
    but also consider the 'wasted capacity' if an item is too large relative
    to remaining capacity. We use a scoring function that balances:
    1. Value density (value / weight)
    2. Feasibility margin (how much of the remaining capacity the item uses)
    3. Absolute value contribution
    
    Specifically, we score each item as:
    score = (value / weight) * (1 + alpha * (value / remaining_capacity))
    
    This boosts items that have high density AND high absolute value relative to capacity.
    
    If no items are feasible (all weights > remaining_capacity), return -1.
    """
    # Convert to numpy arrays for vectorized operations
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Check if any items are feasible
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    # Calculate value density
    density = feasible_values / feasible_weights
    
    # Calculate capacity utilization ratio (how much of capacity this item uses)
    # Items that use a reasonable fraction of capacity are preferred over tiny ones
    # when density is similar
    utilization = feasible_weights / capacity
    
    # Score: combine density with a bonus for higher absolute value
    # The term (1 + beta * value/capacity) boosts high-value items
    beta = 0.5
    score = density * (1.0 + beta * (feasible_values / capacity))
    
    # Additional tie-breaking: prefer items with moderate utilization (not too small, not too large)
    # Penalize very small items that waste capacity in terms of potential future items
    # Actually, for greedy heuristics, high density is usually best. Let's stick with the above.
    
    # Find the best item
    best_local_idx = int(np.argmax(score))
    best_global_idx = int(feasible_indices[best_local_idx])
    
    return best_global_idx
# EVOLVE-BLOCK-END

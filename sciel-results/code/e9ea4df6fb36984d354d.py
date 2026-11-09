"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a multi-objective dominance filter.
    
    Approach:
    1. Identify the most constraining resource for each item (the constraint where
       the item consumes the largest fraction of its capacity).
    2. Compute prize per unit of that most constraining resource for each item.
    3. Apply a Pareto dominance filter: an item is dominated if there exists another
       item with higher prize in ALL constraints AND lower or equal weight in ALL constraints.
       More practically, we use a simplified dominance: an item is dominated if there's
       another item with strictly higher total prize and strictly lower total weight.
    4. Assign high scores to non-dominated (Pareto-optimal) items, scaled by their
       efficiency ratio, and low scores to dominated items.
    5. Add a small noise/bias to break ties and ensure positive values.
    
    Returns an array of shape (100,) with one score per item.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Ensure shapes are valid
    if prizes.shape != weights.shape or prizes.ndim != 2:
        return np.ones(n_items, dtype=np.float64)
    
    # Compute total prize and total weight per item
    total_prize = prizes.sum(axis=0)  # shape (n_items,)
    total_weight = weights.sum(axis=0)  # shape (n_items,)
    
    # Identify the most constraining resource for each item:
    # For each item, find which constraint it uses the largest fraction of capacity (25)
    # capacity = 25 for all constraints
    capacity = 25.0
    # Fraction of capacity used per constraint per item
    fractions = weights / capacity  # shape (n_constraints, n_items)
    # Most constraining resource index per item
    most_constraining = np.argmax(fractions, axis=0)  # shape (n_items,)
    
    # Prize per unit of the most constraining resource
    # For item j, the most constraining resource is c = most_constraining[j]
    # Efficiency = prizes[c, j] / weights[c, j] (if weights[c, j] > 0)
    efficiencies = np.zeros(n_items, dtype=np.float64)
    for j in range(n_items):
        c = most_constraining[j]
        if weights[c, j] > 1e-12:
            efficiencies[j] = prizes[c, j] / weights[c, j]
        else:
            efficiencies[j] = total_prize[j] / (total_weight[j] + 1e-12)
    
    # Apply Pareto dominance filter:
    # An item i is dominated by item k if:
    #   total_prize[k] > total_prize[i] AND total_weight[k] < total_weight[i]
    # (simplified 2D Pareto front)
    dominated = np.zeros(n_items, dtype=bool)
    for i in range(n_items):
        for k in range(n_items):
            if k == i:
                continue
            if total_prize[k] > total_prize[i] + 1e-12 and total_weight[k] < total_weight[i] - 1e-12:
                dominated[i] = True
                break
    
    # Assign scores:
    # Non-dominated items get high scores based on their efficiency
    # Dominated items get low scores
    scores = np.ones(n_items, dtype=np.float64)
    
    # For non-dominated items, scale by efficiency
    non_dominated_mask = ~dominated
    if np.any(non_dominated_mask):
        eff_nd = efficiencies[non_dominated_mask]
        # Normalize efficiencies to [1, 10] range for non-dominated items
        if len(eff_nd) > 1:
            eff_min = np.min(eff_nd)
            eff_max = np.max(eff_nd)
            if eff_max > eff_min + 1e-12:
                normalized = (eff_nd - eff_min) / (eff_max - eff_min)
                scores[non_dominated_mask] = 1.0 + 9.0 * normalized
            else:
                scores[non_dominated_mask] = 5.0
        else:
            scores[non_dominated_mask] = 5.0
    
    # For dominated items, give low scores
    scores[dominated] = 0.1
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

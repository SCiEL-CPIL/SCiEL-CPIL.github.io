"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    # Ensure correct shapes and types
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 2:
        p = prizes
    else:
        p = prizes.reshape(1, -1)
    
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    
    # Calculate per-item metrics
    # p is (1, n_items) or (5, n_items)
    # If p has shape (5, 100), we need a single prize per item for the heuristic
    # The contract says "prizes and weights, of shape (5, 100)". 
    # Wait, the task says "prizes and weights, of shape (5, 100)".
    # Usually in MCKP, prizes are per item (1, n) and weights are (m, n).
    # But here it says prizes is (5, 100). This is unusual.
    # Let's re-read: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # This implies prizes is also (5, 100).
    # However, the scoring is "highest total prize".
    # If prizes is (5, 100), which prize is the "total prize"?
    # Usually, the objective is to maximize sum of prizes. If there are 5 prize vectors,
    # maybe it's the sum across constraints? Or maybe the first row?
    # Let's look at the current program: `p = np.mean(prizes, axis=0)`.
    # This assumes the prize to maximize is the average of the 5 prize rows.
    # Or maybe the prize is just one value per item, and the (5,100) shape is a quirk.
    # Let's assume the "prize" of an item is the sum or mean of its entries in the prize matrix.
    # Given the current code uses mean, and it scored 0.99, let's stick with a similar interpretation
    # but maybe sum is better if we want to maximize total?
    # Actually, if prizes is (5, 100), and we pick a subset of items, the total prize is likely
    # the sum of the selected items' prizes. If each item has 5 prize values, maybe the total prize
    # is the sum of all 5 values for each selected item? Or just one of them?
    # Let's assume the "prize" of an item is the sum of its row values if it's (5,100),
    # or just the value if it's (100,).
    # Let's use the sum across the first axis if 2D, as that represents the total "value" density.
    # However, the current code uses mean. Mean and sum differ by a constant factor (5),
    # which doesn't change the relative order in the heuristic if we normalize.
    # So let's use the sum or mean, it shouldn't matter much for ranking.
    # Let's use the sum to be safe about "total prize".
    
    if p.shape[0] == 1:
        item_prizes = p[0]
    else:
        # Sum across the 5 dimensions
        item_prizes = np.sum(p, axis=0)
    
    # Weights are (5, 100). Capacity is 25 for each of the 5 constraints.
    # An item i fits if for all c in 0..4: sum_{j in S} w[j, c] + w[i, c] <= 25.
    
    # Strategy: Model item attractiveness as expected marginal utility gain.
    # Estimate "shadow price" of each constraint via inverse of its total weight sum.
    # Score items by prize minus weighted sum of resource costs.
    
    # Calculate total weight per constraint
    total_weights = np.sum(weights, axis=1)  # (5,)
    
    # Avoid division by zero
    safe_total_weights = np.where(total_weights > 1e-9, total_weights, 1e-9)
    
    # Shadow prices: inverse of total weight per constraint
    # This represents the "scarcity" or "cost" of each unit of resource in that constraint
    shadow_prices = 1.0 / safe_total_weights  # (5,)
    
    # Compute weighted resource cost for each item
    # cost_i = sum_c (shadow_price_c * weight_i_c)
    costs = np.sum(shadow_prices[:, np.newaxis] * weights, axis=0)  # (100,)
    
    # Attractiveness: prize - costs
    # Items with high prize and low total weighted cost are more attractive
    h = item_prizes - costs
    
    # Ensure positive values (contract requires positive heuristic entries)
    # If h is not positive, set to 1e-9
    h = np.where(h > 1e-9, h, 1e-9)
    
    # Normalize to mean 1 for consistency
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
        
    return h
# EVOLVE-BLOCK-END

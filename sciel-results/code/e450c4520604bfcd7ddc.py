"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute competitive exclusion scores for items in a multiple-constraint knapsack.
    
    Strategy: Simulate a greedy packing process to estimate how "crowded" each item is.
    Items that are selected early in the greedy process (high marginal utility relative
    to remaining capacity) get higher heuristic scores. Items that are only selected
    late (or not at all) because they compete with better items get lower scores.
    
    The heuristic is: prize[i] / (1 + competition[i]) where competition[i] measures
    how many other items with higher prize-to-weight ratios exist in the same constraint
    space, effectively modeling opportunity cost.
    """
    start_time = time.time()
    deadline = 330.0 - 30.0  # 30s safety margin
    
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Compute per-constraint efficiency: prize / weight for each constraint
    # This gives us n_constraints x n_items matrix of efficiencies
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.zeros_like(prizes)
        for c in range(n_constraints):
            w = weights[c]
            mask = w > 1e-9
            eff[c, mask] = prizes[c, mask] / w[mask]
            eff[c, ~mask] = prizes[c, ~mask]  # if weight is 0, efficiency is just prize
    
    # For each item, compute the minimum efficiency across all constraints
    # This represents the "bottleneck" constraint for that item
    min_eff = np.min(eff, axis=0)  # shape: (n_items,)
    
    # Now simulate a greedy packing to determine "competition rank"
    # Sort items by min_eff descending
    order = np.argsort(-min_eff)
    
    # Track remaining capacity
    remaining = np.full(n_constraints, capacity)
    
    # competition[i] = number of items selected before i that share tight constraints
    competition = np.zeros(n_items)
    
    # We'll do a simplified simulation:
    # For each item in greedy order, if it fits, select it and reduce capacity.
    # Items that don't fit get high competition scores.
    
    selected = np.zeros(n_items, dtype=bool)
    
    # To make this more informative, we'll track for each item how "tight" its selection was
    # and how many other items it "blocks"
    
    # Alternative approach: compute a "crowding" score
    # For each constraint c, sort items by weight descending
    # Items with high weight in a constraint "crowd out" others
    
    # Let's use a different competitive exclusion model:
    # For each item i, compute:
    # score[i] = sum_c (prize[i,c] / (sum_j weights[j,c] for j that are "similar" to i in constraint c))
    
    # Simpler and faster: Use the ratio of item's prize to the total prize of items
    # that would be "competing" for the same constraint capacity
    
    # For each constraint c, the total weight is sum(weights[c, :])
    # The "density" of constraint c is total_weight / capacity
    # Items in dense constraints have higher opportunity cost
    
    # Compute constraint density
    total_weight = weights.sum(axis=1)  # shape: (n_constraints,)
    density = total_weight / capacity  # shape: (n_constraints,)
    
    # For each item, compute a weighted average of per-constraint "scarcity-adjusted" efficiency
    # Scarcity-adjusted efficiency: eff[c, i] / density[c]
    # This penalizes items in constraints that are already heavily utilized
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity_adj_eff = eff / density[np.newaxis, :]  # shape: (n_constraints, n_items)
    
    # The heuristic is the geometric mean (harmonic mean) of scarcity-adjusted efficiencies
    # across constraints, which penalizes items that are inefficient in any tight constraint
    
    # Use harmonic mean: n_constraints / sum(1/x)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sum = np.sum(1.0 / (scarcity_adj_eff + 1e-9), axis=0)
        harmonic_mean = n_constraints / (inv_sum + 1e-9)
    
    # Ensure positivity
    harmonic_mean = np.maximum(harmonic_mean, 1e-9)
    
    # Add a competitive exclusion component:
    # For each item, penalize based on how many other items have higher efficiency
    # in the same bottleneck constraint
    
    # Find the bottleneck constraint for each item
    bottleneck = np.argmin(eff, axis=0)  # shape: (n_items,)
    
    # For each item, count how many other items have higher efficiency in its bottleneck constraint
    # This is expensive (O(n^2)), so we'll approximate using rank
    
    # For each constraint c, compute the rank of each item by efficiency
    ranks = np.zeros_like(eff)
    for c in range(n_constraints):
        order_c = np.argsort(-eff[c])
        rank_arr = np.empty(n_items)
        rank_arr[order_c] = np.arange(n_items)
        ranks[c] = rank_arr
    
    # For each item, get its rank in its bottleneck constraint
    item_ranks = ranks[range(n_constraints), range(n_items)]  # This is wrong indexing
    
    # Correct indexing:
    item_ranks = np.zeros(n_items)
    for i in range(n_items):
        item_ranks[i] = ranks[bottleneck[i], i]
    
    # Competition score: higher rank (worse) means more competition
    # Normalize ranks to [0, 1]
    max_rank = n_items - 1
    competition_norm = item_ranks / (max_rank + 1e-9)
    
    # Final heuristic: harmonic_mean * (1 - competition_norm)^2
    # This strongly penalizes items that are poor in their bottleneck constraint
    
    heuristic = harmonic_mean * (1.0 - competition_norm) ** 2
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a simple fallback
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

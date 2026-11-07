"""Rewritten program for ant colony bin packing heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values indicate more attractive placements.
    
    Strategy: We want to encourage placements that maximize the number of future
    items that can still fit in the bin after placing j after i. This is a
    direct measure of how "good" the placement is for subsequent packing.
    
    For each pair (i, j) where demands[i] + demands[j] <= capacity:
      - Count how many other items k (k != i, k != j) can still fit after
        placing i and j in the same bin.
      - The heuristic value is proportional to this count.
    
    This directly optimizes for the goal of fitting more items per bin.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Feasibility: pair must fit
    feasible = pair_sum <= capacity
    
    # Residual capacity after placing i and j
    residual = capacity - pair_sum
    
    # For each pair (i, j), count how many items k can still fit
    # Item k can fit if demands[k] <= residual
    # We need to exclude i and j themselves from the count
    
    # Create a matrix where fit[i, j, k] = 1 if demands[k] <= residual[i, j]
    # But we can't easily make a 3D array efficiently for n=120, so let's think differently.
    
    # For each pair (i, j), the number of items that can fit is:
    # count of k where demands[k] <= residual[i, j]
    # But we should exclude i and j from this count
    
    # Let's precompute for each possible residual value, how many items fit
    # Since demands are integers from 20 to 99, residual can range from 0 to 110
    
    # Count how many items have demand <= x for x in range
    # demands are floats but originally integers from 20 to 99
    # Let's use np.searchsorted on sorted demands
    
    sorted_demands = np.sort(demands)
    
    # For each pair (i, j), count how many items fit in the residual
    # We want count of k != i, k != j where demands[k] <= residual[i, j]
    
    # First, count all items where demand <= residual
    # Then subtract 1 if demands[i] <= residual[i, j] (since i is already placed)
    # And subtract 1 if demands[j] <= residual[i, j] (since j is already placed)
    
    # For each pair (i, j), the number of items with demand <= residual[i, j]
    # Use np.searchsorted on sorted_demands
    # np.searchsorted(sorted_demands, residual, side='right') gives count of elements <= residual
    
    # But residual is a 2D array, so we need to do this efficiently
    # Let's use broadcasting
    
    # count_all[i, j] = number of items k where demands[k] <= residual[i, j]
    # We can compute this by comparing each demand to each residual
    
    # demands[k] <= residual[i, j] for all i, j, k
    # This is a 3D comparison, which is expensive for n=120
    
    # Alternative: for each unique residual value, compute the count once
    # But residuals are continuous, so let's just do the 3D approach carefully
    
    # Actually, let's use a different approach:
    # For each item k, it can fit after (i, j) if demands[k] <= residual[i, j]
    # So for each k, the set of (i, j) where k fits is where residual[i, j] >= demands[k]
    
    # Let's build the heuristic by summing over k
    # heuristic[i, j] = sum over k of [demands[k] <= residual[i, j]] - [i == k] - [j == k]
    
    # This is equivalent to:
    # For each k, add 1 to all (i, j) where residual[i, j] >= demands[k]
    # Then subtract 1 from (k, j) for all j (since i=k)
    # And subtract 1 from (i, k) for all i (since j=k)
    
    # But we only care about feasible (i, j) pairs
    
    # Let's build the matrix
    heuristic = np.zeros((n, n), dtype=np.float64)
    
    # For each item k, find all (i, j) where residual[i, j] >= demands[k]
    # residual[i, j] = capacity - demands[i] - demands[j]
    # So we need capacity - demands[i] - demands[j] >= demands[k]
    # i.e., demands[i] + demands[j] <= capacity - demands[k]
    
    for k in range(n):
        threshold = capacity - demands[k]
        # Find all (i, j) where demands[i] + demands[j] <= threshold
        # This is a pairwise sum comparison
        pair_sum_k = demands[:, None] + demands[None, :]
        mask = pair_sum_k <= threshold
        heuristic += mask
    
    # Now subtract the self-exclusions
    # For each k, we added 1 to (i, j) for all i, j where the pair fits with k
    # But we need to exclude i=k and j=k
    # For i=k: subtract 1 from (k, j) for all j where (k, j) was counted
    # For j=k: subtract 1 from (i, k) for all i where (i, k) was counted
    
    # Actually, let's reconsider. The heuristic should count how many OTHER items
    # can fit after placing i and j. So for pair (i, j), we count k != i, k != j
    # where demands[k] <= residual[i, j].
    
    # My loop above adds 1 for each k where demands[i] + demands[j] <= capacity - demands[k]
    # which is equivalent to demands[k] <= residual[i, j]
    # But this includes k=i and k=j if they satisfy the condition.
    
    # So we need to subtract:
    # For pair (i, j), subtract 1 if demands[i] <= residual[i, j] (i.e., 2*demands[i] + demands[j] <= capacity)
    # And subtract 1 if demands[j] <= residual[i, j] (i.e., demands[i] + 2*demands[j] <= capacity)
    
    # Let's compute these corrections
    # For each pair (i, j), check if i can fit: demands[i] <= residual[i, j]
    # i.e., demands[i] <= capacity - demands[i] - demands[j]
    # i.e., 2*demands[i] + demands[j] <= capacity
    
    self_i = 2 * demands[:, None] + demands[None, :] <= capacity
    self_j = demands[:, None] + 2 * demands[None, :] <= capacity
    
    heuristic -= self_i.astype(np.float64)
    heuristic -= self_j.astype(np.float64)
    
    # Ensure non-negative
    heuristic = np.maximum(heuristic, 0)
    
    # Apply feasibility mask
    heuristic = np.where(feasible, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

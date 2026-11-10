"""Improved heuristic for ant colony offline bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)

    # Pairwise sum of item sizes
    pair = d[:, None] + d[None, :]

    # A pair is "feasible" if their sum <= capacity
    feasible = pair <= cap

    # Remaining space after placing both items
    remaining = cap - pair

    # We want to favor pairs that leave MORE remaining space (i.e., smaller pair sum)
    # but also ensure feasibility. For infeasible pairs, set to a tiny value.
    
    # Use a smooth score: for feasible pairs, score = 1 / (pair_sum + epsilon)
    # This rewards smaller sums more strongly.
    # For infeasible pairs, score = 1e-9
    
    eps = 1e-6
    scores = np.where(feasible, 1.0 / (pair + eps), 1e-9)

    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)

    return scores
# EVOLVE-BLOCK-END

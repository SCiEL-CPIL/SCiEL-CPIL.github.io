"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic is formulated as a static, instance-specific 'opportunity cost' matrix.
    Each entry H[i, j] represents the attractiveness of moving from node i to node j.
    
    The heuristic combines:
    1. The prize value of the destination node j (higher prize = more attractive).
    2. A penalty based on how much of the remaining length budget is consumed by the
       trip i -> j -> 0, explicitly penalizing nodes that are too far relative to their value.
    3. A penalty for the distance from i to j itself.
    
    Specifically:
    - For each pair (i, j), compute the 'round-trip cost' of going from i to j and then
      back to the depot (node 0): cost_ij = d[i, j] + d[j, 0].
    - The fraction of the max_len budget consumed is cost_ij / max_len.
    - We want to prioritize nodes where this fraction is small relative to the prize.
    - The heuristic is defined as: H[i, j] = prize[j] / (cost_ij + epsilon)^alpha
      where alpha controls the strength of the distance penalty.
    
    However, we must be careful: the ant's current position is i, and it has already
    traveled some distance. But since the heuristic is computed once before the colony
    runs, we don't know the current path length. So we use a static approximation:
    assume the ant has used 0 length so far, and the constraint is that d[i,j] + d[j,0]
    must be <= max_len for the move to be feasible at all (from the start). But actually,
    the constraint depends on the path so far. Since the heuristic is static, we can only
    encode a static notion of 'goodness'.
    
    A better static heuristic: for each edge (i, j), compute how 'expensive' it is in terms
    of the budget it consumes relative to the prize gained. The opportunity cost is:
    (d[i, j] + d[j, 0]) / max_len  -- the fraction of the total budget that this move+return
    would consume if starting fresh from i. But since i is not necessarily 0, this isn't
    quite right either.
    
    Let's think differently. The key insight from the directive: "the prize of node j multiplied
    by the fraction of the remaining length budget that would be consumed by visiting it".
    
    Wait, re-reading: "each entry is the prize of node j multiplied by the fraction of the
    remaining length budget that would be consumed by visiting it". That seems backwards - 
    we'd want to multiply by (1 - fraction) to reward nodes that consume little budget.
    
    Actually, let me re-interpret: the heuristic should be high when prize[j] is high AND
    the cost of visiting j is low. So:
    
    H[i, j] = prize[j] * (1 - (d[i,j] + d[j,0]) / max_len) for feasible edges, and small
    for infeasible ones. But this can be negative.
    
    Better: H[i, j] = prize[j] / (d[i,j] + d[j,0] + epsilon) ** beta
    
    Let's use a power-law with beta = 1.0 or 2.0.
    
    Actually, the current seed uses (prizes + eps) / (distance + eps). Let's enhance this
    by incorporating the return distance d[j, 0] as well, since the ant must eventually
    return to 0.
    
    H[i, j] = prize[j] / ((d[i, j] + d[j, 0]) + epsilon) ** 1.5
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # For each destination j, the cost of visiting j from i and returning to 0 is:
    # d[i, j] + d[j, 0]
    # We want to penalize this cost.
    
    # Compute the return cost for each j: d[j, 0]
    return_cost = distance_matrix[:, 0]  # shape (n,)
    
    # For each (i, j), total cost = d[i, j] + d[j, 0]
    # This is distance_matrix + return_cost[None, :]
    total_cost = distance_matrix + return_cost[None, :]  # shape (n, n)
    
    # Heuristic: prize[j] / (total_cost + eps) ** 1.5
    # This rewards high-prize nodes that are close in terms of round-trip cost.
    
    # Clip total_cost to avoid division by zero or negative values
    total_cost_safe = np.maximum(total_cost, eps)
    
    heuristic = prizes[None, :] / (total_cost_safe ** 1.5)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

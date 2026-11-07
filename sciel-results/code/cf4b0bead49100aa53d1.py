"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Formulated as a static, instance-specific "opportunity cost" matrix where each entry
    is the prize of node j multiplied by the fraction of the remaining length budget that
    would be consumed by visiting it, thereby explicitly penalizing nodes that are too far
    relative to their value and forcing the ACO to prioritize high-value nodes that leave
    sufficient slack for the return trip.
    
    H[i, j] = prize[j] * (1 - (d[i, j] + d[j, 0]) / max_len) for feasible edges,
    with a floor to keep values positive.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # For each destination j, the cost of visiting j from i and returning to 0 is:
    # d[i, j] + d[j, 0]
    return_cost = distance_matrix[:, 0]  # shape (n,)
    
    # For each (i, j), total cost = d[i, j] + d[j, 0]
    total_cost = distance_matrix + return_cost[None, :]  # shape (n, n)
    
    # Fraction of budget consumed
    budget_fraction = total_cost / max_len
    
    # Remaining budget fraction: 1 - budget_fraction
    # Clip to [0, 1] to handle infeasible edges (where budget_fraction > 1)
    remaining_fraction = np.clip(1.0 - budget_fraction, 0.0, 1.0)
    
    # Heuristic: prize[j] * remaining_fraction
    # This is high when prize is high AND the move consumes little of the budget
    heuristic = prizes[None, :] * remaining_fraction
    
    # Ensure all values are positive and finite
    # For infeasible edges or zero-prize nodes, use a small positive value
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Build an edge-attractiveness matrix using a structural flow-based approach.
    
    Strategy: Use the Hungarian algorithm to find a minimum-cost cycle cover
    (assignment) on the complete graph, which approximates a set of non-overlapping
    TSP tours. This encodes the global routing structure. Then combine with
    Clarke-Wright savings and demand-aware adjustments.
    
    The cycle cover approach is structurally different from pure distance-based
    heuristics: it finds edges that are part of globally coherent tours rather
    than just locally short edges.
    """
    n = distance_matrix.shape[0]
    d = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    capacity = float(capacity)
    
    # Ensure finite distances
    d = np.where(np.isfinite(d), d, 1e6)
    d = np.maximum(d, 1e-9)
    
    # 1. Find minimum-cost cycle cover using Hungarian algorithm
    # This gives us a set of edges that form non-overlapping cycles covering all nodes
    # These edges are structurally important for routing
    cost_matrix = d.copy()
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    # Create a binary matrix of edges in the cycle cover
    cycle_cover = np.zeros((n, n), dtype=np.float64)
    for i, j in zip(row_idx, col_idx):
        if i != j:  # Exclude self-loops
            cycle_cover[i, j] = 1.0
    
    # 2. Base distance heuristic: 1 / (d^2 + eps)
    eps = 1e-4
    dist_heur = 1.0 / (d**2 + eps)
    
    # 3. Clarke-Wright savings: S[i,j] = d[i,0] + d[0,j] - d[i,j]
    d_i0 = d[:, 0].reshape(-1, 1)  # (n, 1)
    d_0j = d[0, :].reshape(1, -1)  # (1, n)
    
    S = d_i0 + d_0j - d
    S = np.maximum(S, 0.0)
    
    # Normalize savings to [0, 1]
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # 4. Demand-aware adjustment
    gamma = 0.5
    demand_penalty = 1.0 / (1.0 + gamma * (demands / capacity))
    demand_penalty[0] = 1.0  # depot has no demand penalty
    demand_mult = demand_penalty[np.newaxis, :]  # (1, n)
    
    # 5. Combine all components
    # The cycle cover boost emphasizes edges that are part of globally coherent tours
    # lambda_cc: strength of cycle cover boost
    # lambda_s: strength of savings boost
    lambda_cc = 2.0
    lambda_s = 1.0
    
    cycle_cover_mult = 1.0 + lambda_cc * cycle_cover
    savings_mult = 1.0 + lambda_s * S_norm
    
    heuristic = dist_heur * cycle_cover_mult * savings_mult * demand_mult
    
    # 6. Handle diagonal and invalid values
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are finite and positive (or 1e-9 as per contract)
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END

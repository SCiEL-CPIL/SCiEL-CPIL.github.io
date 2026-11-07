"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base attractiveness: inverse distance
    base = 1.0 / (distance_matrix + eps)
    
    # Compute "essentiality" of each edge: how much it's needed vs shortcuts
    # For each pair (i,j), find min over k != i,j of d_ik + d_kj
    # If d_ij is close to this min, the edge is essential
    # If d_ij is much less, it's a valuable shortcut
    
    # Actually, by triangle inequality, d_ij <= d_ik + d_kj
    # So the ratio r_ij = d_ij / min_k(d_ik + d_kj) is in (0, 1]
    # High r means the edge is "necessary" (no good shortcut)
    # Low r means there's a good shortcut, so this edge is less critical
    
    # But for ACO, we want to encourage edges that lead to good routes
    # Edges with high r are "hard to replace" - they might be in the MST
    # Edges with low r have alternatives
    
    # Let's use: heuristic = base * (r^0.5 + 0.5) to boost essential edges
    # r^0.5 softens the effect
    
    # Compute min_k(d_ik + d_kj) for all i,j
    # This is O(n^3)
    min_via = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                min_via[i, j] = np.inf
                continue
            # min over k != i, j of d_ik + d_kj
            vals = distance_matrix[i, :] + distance_matrix[:, j]
            vals[i] = np.inf
            vals[j] = np.inf
            min_via[i, j] = np.min(vals)
    
    # Ratio r_ij = d_ij / min_via_ij
    # Handle division by zero/inf
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = distance_matrix / min_via
    
    # For i==j, set ratio to 1 (won't be used)
    np.fill_diagonal(ratio, 1.0)
    
    # Clip ratio to [0, 1] (should be by triangle inequality, but numerical issues)
    ratio = np.clip(ratio, 0.0, 1.0)
    
    # Boost factor: edges with high ratio (essential) get boosted
    # Use sqrt to soften
    boost = np.sqrt(ratio) + 0.5
    
    heuristic = base * boost
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

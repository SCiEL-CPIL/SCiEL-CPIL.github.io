"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a combination of:
    1. Minimum Spanning Tree (MST) edges: high attractiveness to enforce connectivity
    2. Non-MST edges: inversely proportional to distance for geometric locality
    
    The heuristic is a linear combination that favors MST edges while maintaining
    distance-based locality for all edges.
    """
    n = distance_matrix.shape[0]
    
    # Handle edge cases
    if n <= 1:
        return np.ones_like(distance_matrix, dtype=float)
    
    # Ensure distance matrix is symmetric and has zero diagonal
    dist = np.array(distance_matrix, dtype=float, copy=True)
    
    # Make symmetric (take minimum of both directions to be safe)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    
    # Replace any non-finite or negative values with large penalty
    dist = np.where(np.isfinite(dist), dist, 1e6)
    dist = np.where(dist > 0, dist, 1e-9)
    
    # Compute MST
    try:
        mst_graph = minimum_spanning_tree(dist, directed=False)
        mst_matrix = mst_graph.toarray()
        # MST edges have weight = distance, non-MST edges have inf
        # Create binary mask for MST edges
        mst_mask = np.isfinite(mst_matrix) & (mst_matrix > 0)
        # Symmetrize the mask
        mst_mask = mst_mask | mst_mask.T
        np.fill_diagonal(mst_mask, False)
    except Exception:
        # Fallback: just use inverse distance
        mst_mask = np.zeros_like(dist, dtype=bool)
    
    # Base heuristic: inverse distance raised to a power to emphasize locality
    # p=2.0 provides a stronger gradient than p=1.0, discouraging long jumps
    p = 2.0
    base_heuristic = 1.0 / (dist + 1e-9) ** p
    
    # Handle diagonal and non-finite values in base heuristic
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 0.0)
    np.fill_diagonal(base_heuristic, 0.0)
    
    # MST edge boost: multiply by a factor to strongly encourage MST edges
    # This creates a linear combination: H = base + (base * boost) * I_mst
    # Effectively: H = base * (1 + boost * I_mst)
    boost_factor = 3.0
    mst_multiplier = 1.0 + boost_factor * mst_mask.astype(float)
    
    # Final heuristic
    heuristic = base_heuristic * mst_multiplier
    
    # Ensure all values are positive and finite
    # The problem states non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Zero out diagonal (no self-loops) - ensure min value
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

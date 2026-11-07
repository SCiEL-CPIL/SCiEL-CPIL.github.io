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
    
    # Base heuristic: inverse distance (closer = more attractive)
    # Use a softened inverse to avoid extreme values
    base_heuristic = 1.0 / (dist + 1e-9)
    
    # Normalize base heuristic to [0, 1] range for better control
    max_base = np.max(base_heuristic)
    min_base = np.min(base_heuristic)
    if max_base > min_base:
        base_norm = (base_heuristic - min_base) / (max_base - min_base)
    else:
        base_norm = np.ones_like(base_heuristic)
    
    # MST edge bonus: add extra attractiveness to MST edges
    # This encourages ants to use MST edges, promoting connectivity
    mst_bonus = np.where(mst_mask, 1.0, 0.0)
    
    # Combine: weight the base heuristic and add MST bonus
    # alpha controls the relative importance of MST edges vs. distance locality
    alpha = 2.0  # MST edges get 2x the normalized base heuristic boost
    
    # Final heuristic: base (normalized) + MST bonus
    # Scale so that typical values are around 1-3
    heuristic = (1.0 + alpha * base_norm) + mst_bonus * (1.0 + alpha * base_norm)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

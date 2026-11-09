"""Guided local search penalty for TSP; Rank-based convex usage penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a rank-based convex penalty to edges based on their usage count.
    
    Strategy:
    Instead of targeting specific edges in the current tour (which can be fragile if the
    tour is not a true local optimum on the true distances), we apply a global penalty
    to ALL edges in the matrix. The penalty magnitude is determined by the edge's rank
    in the distribution of usage counts. Specifically, we use the square of the normalized
    rank. This creates a convex penalty landscape:
    - Edges with very low usage (low rank) get almost no penalty.
    - Edges with moderate usage get a moderate penalty.
    - Edges with very high usage (high rank, i.e., most frequently used) get a very strong penalty.
    
    This focuses the search's disruptive power on the most entrenched local minima structures
    (which are built from the most frequently used edges) while preserving the geometry of
    less-explored regions. This is a structural change from the current "trap detection"
    approach, moving from a local geometric heuristic to a global statistical penalty.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    D = np.asarray(edge_distance, dtype=float)
    N = np.asarray(edge_n_used, dtype=float)
    
    # Symmetrize inputs
    D = (D + D.T) / 2.0
    N = (N + N.T) / 2.0
    
    n = D.shape[0]
    if n < 4:
        return D.copy()
        
    # Check time budget
    if time.time() > deadline:
        return D.copy()
        
    # Flatten the upper triangle of N to get a list of usage counts for unique edges
    # We only need the ranks, so we can work with the flattened array
    i_upper, j_upper = np.triu_indices(n, k=1)
    counts = N[i_upper, j_upper]
    
    # If there are no used edges yet, return original distances
    if np.all(counts == 0):
        return D.copy()
        
    # Calculate ranks of the usage counts
    # We want edges with high usage to have high rank (close to 1.0)
    # np.argsort(np.argsort(counts)) gives the rank of each element
    # For ties, this is arbitrary but consistent.
    ranks = np.argsort(np.argsort(counts)).astype(float)
    
    # Normalize ranks to [0, 1]
    # If max rank is 0 (all counts same), avoid division by zero
    max_rank = np.max(ranks)
    if max_rank == 0:
        return D.copy()
        
    normalized_ranks = ranks / max_rank
    
    # Apply convex penalty: square of normalized rank
    # This makes the penalty grow quadratically for high-usage edges
    penalty_magnitudes = normalized_ranks ** 2
    
    # Scale the penalty. 
    # We want the penalty for the most used edges to be significant but not infinite.
    # Let's scale it relative to the average distance or max distance.
    # Using max_dist * factor is robust.
    max_dist = np.max(D)
    # A factor of 1.0 means the most used edge gets an added distance of max_dist.
    # This effectively doubles its length, making 2-opt very likely to remove it.
    # We can tune this factor. 1.0 is a strong kick.
    penalty_scale = max_dist * 1.0
    
    # Add the penalty to the distance matrix
    # We need to map the 1D penalty array back to the 2D matrix
    new_dist = D.copy()
    new_dist[i_upper, j_upper] += penalty_magnitudes * penalty_scale
    new_dist[j_upper, i_upper] += penalty_magnitudes * penalty_scale
    
    # Ensure diagonal is 0
    np.fill_diagonal(new_dist, 0.0)
    
    # Safety check
    if not np.all(np.isfinite(new_dist)):
        return D.copy()
        
    return new_dist
# EVOLVE-BLOCK-END

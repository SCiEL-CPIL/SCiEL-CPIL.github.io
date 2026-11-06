"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    We penalize edges that have been frequently used in previous local optima,
    with a penalty that grows with usage but saturates to avoid over-penalizing.
    We also add a small immediate penalty to edges in the current tour to encourage
    escape from the current local optimum.
    
    Parameters:
    -----------
    edge_distance : np.ndarray
        True Euclidean distance matrix (symmetric, shape (n, n)).
    local_opt_tour : list or np.ndarray
        The current local optimum tour as a sequence of node indices.
    edge_n_used : np.ndarray
        Symmetric count matrix of how many times each edge has been used
        in previous local optima.
        
    Returns:
    --------
    np.ndarray
        Perturbed distance matrix with same shape as edge_distance.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s kill
    
    n = edge_distance.shape[0]
    d_out = edge_distance.copy()
    
    if n < 2:
        return d_out
    
    # Extract edges from the current tour
    # local_opt_tour is a sequence of nodes; edges are (tour[i], tour[(i+1)%n])
    tour = list(local_opt_tour)
    tour_len = len(tour)
    
    if tour_len < 2:
        return d_out
    
    # Build set of edges in current tour
    current_edges = set()
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        if u != v:
            current_edges.add((min(u, v), max(u, v)))
    
    # Compute penalty parameters
    # Use a saturating penalty: penalty = alpha * count / (1 + count)
    # This ensures the penalty is bounded and doesn't grow unboundedly.
    
    # Base penalty for edges in current tour (immediate escape pressure)
    alpha_current = 0.3
    
    # Penalty for previously used edges (cumulative memory)
    alpha_used = 0.5
    
    # Apply penalties
    for (u, v) in current_edges:
        d_out[u, v] += alpha_current
        d_out[v, u] += alpha_current
    
    # Apply cumulative penalty for all edges that have been used before
    # Use a saturating function: penalty = alpha_used * n_used / (1 + n_used)
    # This is equivalent to alpha_used * (1 - 1/(1+n_used))
    # We apply this to all edges where edge_n_used > 0
    
    # Vectorized computation for efficiency
    # Only modify off-diagonal elements where edge_n_used > 0
    mask = (edge_n_used > 0) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    
    if np.any(mask):
        # Compute saturating penalty
        counts = edge_n_used[mask].astype(float)
        penalties = alpha_used * counts / (1.0 + counts)
        
        # Apply penalties
        d_out[mask] += penalties
    
    # Ensure the matrix is symmetric
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure all values are finite and non-negative
    d_out = np.where(np.isfinite(d_out), d_out, 0.0)
    d_out = np.maximum(d_out, 0.0)
    
    return d_out
# EVOLVE-BLOCK-END

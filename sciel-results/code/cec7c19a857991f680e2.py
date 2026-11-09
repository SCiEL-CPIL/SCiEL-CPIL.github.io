"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Strategy: Combine inverse distance (short edges are attractive) with
    a demand-aware capacity penalty. Edges between nodes with high combined
    demand relative to capacity get slightly penalized, while edges that
    are short and feasible get boosted. We also add a small spectral component
    that identifies natural cluster boundaries via the Fiedler vector.
    
    The heuristic is computed once per instance, before the colony runs.
    """
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance (closer = more attractive)
    # Add small epsilon to avoid division by zero
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-aware penalty: if two customers together exceed capacity,
    # they can't be on the same route consecutively in a useful way.
    # Actually, any two customers can be consecutive as long as the total
    # route load doesn't exceed capacity. The constraint is on the whole route.
    # But we can use a softer signal: prefer edges where the two endpoints
    # have lower individual demands (more flexibility in routing).
    
    # Compute a demand compatibility factor
    # For each edge (i, j), the "remaining capacity" after visiting both
    # is capacity - demands[i] - demands[j]. Higher remaining = more flexible.
    # We use this to slightly boost edges between low-demand nodes.
    demand_sum = demands[:, None] + demands[None, :]
    remaining_cap = capacity - demand_sum
    # Normalize: remaining_cap ranges from capacity - 18 to capacity
    # Make a factor that's 1.0 when remaining_cap is high, slightly less when low
    # Use a smooth sigmoid-like scaling
    demand_factor = 1.0 + 0.1 * (remaining_cap / capacity)
    demand_factor = np.clip(demand_factor, 0.9, 1.1)
    
    # Spectral component: compute Fiedler vector to identify cluster boundaries
    # Build a sparse graph from k-nearest neighbors for efficiency
    k = 5
    # Compute distance-based weights for kNN graph
    knn_weights = np.zeros((n, n))
    for i in range(n):
        dists_i = distance_matrix[i].copy()
        dists_i[i] = np.inf
        nn_idx = np.argsort(dists_i)[:k]
        for j in nn_idx:
            knn_weights[i, j] = 1.0 / (dists_i[j] + 1e-9)
    
    # Symmetrize
    knn_weights = np.maximum(knn_weights, knn_weights.T)
    
    # Compute degree matrix
    degree = knn_weights.sum(axis=1)
    # Avoid division by zero
    degree_safe = np.where(degree > 1e-12, degree, 1.0)
    
    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
    L_sym = np.eye(n) - (knn_weights * d_inv_sqrt[:, None] * d_inv_sqrt[None, :])
    
    # Find Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        # Second smallest eigenvalue index is 1 (0 is the trivial one)
        fiedler = eigenvectors[:, 1]
    except Exception:
        fiedler = np.ones(n) / np.sqrt(n)
    
    # Spectral proximity: edges where both endpoints have similar Fiedler values
    # are within the same "cluster" and should be more attractive
    fiedler_diff_sq = (fiedler[:, None] - fiedler[None, :]) ** 2
    # Normalize: max diff is about 2 (if values in [-1,1])
    spectral_factor = 1.0 + 0.15 * np.exp(-fiedler_diff_sq / 0.1)
    
    # Combine all factors
    heuristic = base_heuristic * demand_factor * spectral_factor
    
    # Ensure non-negative and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

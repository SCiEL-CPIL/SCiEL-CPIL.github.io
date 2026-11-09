"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, QhullError
import numpy.linalg as la


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP.
    
    Approach: Voronoi-based topological heuristic
    - Compute Voronoi diagram to identify spatial regions
    - High attractiveness for edges that connect points in different Voronoi regions
      (bridges between regions)
    - Low attractiveness for edges within the same local cluster
    - Combine with distance-based decay
    
    Returns: matrix of same shape as distance_matrix, larger = more attractive
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Extract coordinates from distance matrix
    # The distance matrix is for 50 uniform random points in unit square
    # We need to recover approximate coordinates or work directly with distances
    
    # Since we only have the distance matrix, we'll use a different strategy:
    # Use the distance matrix itself to identify structure
    
    # Strategy: 
    # 1. For each node, find its k-nearest neighbors
    # 2. Identify "bridge" edges: edges that connect two dense clusters
    # 3. Use a combination of distance and local density
    
    # Compute local density: for each node, the average distance to its k nearest neighbors
    k = min(5, n - 1)
    # Sort distances for each node (excluding self)
    dists = distance_matrix.copy()
    np.fill_diagonal(dists, np.inf)
    
    # For each node, get k nearest neighbors
    knn_dists = np.zeros((n, k))
    for i in range(n):
        sorted_d = np.sort(dists[i])
        knn_dists[i] = sorted_d[:k]
    
    # Local density: average distance to k nearest neighbors
    # Lower density means more isolated point
    local_density = knn_dists.mean(axis=1)
    
    # Normalize local density
    min_dens = local_density.min()
    max_dens = local_density.max()
    if max_dens > min_dens:
        norm_density = (local_density - min_dens) / (max_dens - min_dens)
    else:
        norm_density = np.ones(n) * 0.5
    
    # Identify bridge edges: edges connecting nodes with different local densities
    # An edge (i,j) is a "bridge" if |density[i] - density[j]| is large
    density_diff = np.abs(norm_density[:, np.newaxis] - norm_density[np.newaxis, :])
    
    # Also consider the relative length of the edge compared to typical local distances
    # An edge is "long" if it's much longer than the typical kNN distance of its endpoints
    typical_local = (local_density[:, np.newaxis] + local_density[np.newaxis, :]) / 2
    # Ratio of edge length to typical local distance
    with np.errstate(divide='ignore', invalid='ignore'):
        edge_ratio = distance_matrix / (typical_local + 1e-9)
    
    # Heuristic components:
    # 1. Base distance-based: shorter edges are more attractive
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Bridge bonus: edges connecting regions with different densities get boosted
    #    This encourages the tour to traverse between clusters
    bridge_factor = 1.0 + 2.0 * density_diff
    
    # 3. Long-edge penalty: very long edges relative to local scale are penalized
    #    But we don't want to penalize too aggressively
    long_penalty = np.where(edge_ratio > 3.0, 0.5, 1.0)
    
    # Combine all factors
    heuristic = base_heuristic * bridge_factor * long_penalty
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Symmetrize (should already be symmetric but ensure)
    heuristic = (heuristic + heuristic.T) / 2
    
    return heuristic
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a demand-aware, multi-criteria edge-attractiveness matrix for ACO CVRP.
    
    Combines:
    1. Inverse distance (stronger for closer edges)
    2. Demand penalty: penalize edges to high-demand nodes to encourage packing
       heavy customers together, reducing the number of trips.
    3. Spatial coherence: use a smooth Gaussian kernel based on coordinate proximity
       to encourage spatially coherent routes (reduces backtracking).
    
    The heuristic is eta[i,j] = d(i,j)^-2 * (1/(1 + demand[j])) * spatial_kernel(i,j)
    where spatial_kernel is a Gaussian of the angular deviation or direct coordinate
    distance, encouraging routes that stay in local neighborhoods.
    """
    n = distance_matrix.shape[0]
    
    # Inverse distance squared (stronger distance bias)
    d_safe = np.maximum(distance_matrix, 1e-9)
    inv_dist_sq = 1.0 / (d_safe ** 2)
    
    # Demand-aware penalty: penalize going to high-demand nodes
    # This encourages the ant to visit high-demand nodes when capacity is fresh,
    # packing them into fewer trips
    demand = np.asarray(demands, dtype=np.float64)
    # demand_penalty[j] = 1 / (1 + demand[j])
    # For the depot (node 0), demand is 0, so penalty = 1.0
    demand_penalty = 1.0 / (1.0 + demand[np.newaxis, :])
    
    # Spatial coherence kernel: encourage edges that are spatially coherent
    # Use a Gaussian kernel based on the distance between the two nodes' coordinates
    # relative to the average inter-point distance
    coords = np.asarray(coordinates, dtype=np.float64)
    # Compute pairwise coordinate distances
    coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    coord_dist = np.sqrt(np.sum(coord_diff ** 2, axis=2))
    
    # Average coordinate distance for scaling
    avg_coord_dist = np.mean(coord_dist[coord_dist > 1e-9]) if np.any(coord_dist > 1e-9) else 1.0
    
    # Gaussian kernel: edges that are shorter in coordinate space get higher weight
    # This encourages spatially coherent routes
    sigma = avg_coord_dist * 0.5
    spatial_kernel = np.exp(-(coord_dist ** 2) / (2 * sigma ** 2))
    
    # Combine all criteria
    eta = inv_dist_sq * demand_penalty * spatial_kernel
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END

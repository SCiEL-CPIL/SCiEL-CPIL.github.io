import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    of capacitated vehicle routing.
    
    Uses a multi-scale spatial heuristic combining:
    1. Inverse distance for local connectivity
    2. Demand-weighted centrality for route potential
    3. Spatial density for cluster identification
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base: inverse distance with small epsilon to avoid division by zero
    eps = 1e-9
    inv_dist = 1.0 / (distance_matrix + eps)
    
    # Compute demand-weighted centrality for each node
    # This measures how well-connected a node is in the demand network
    # Higher centrality = node is in a dense, demand-rich region
    demand_weights = demands.copy()
    demand_weights[0] = 0.0  # Depot has no demand
    
    # For each node j, compute its centrality as sum of demand_i / distance(i,j)
    # This identifies nodes that are close to many high-demand customers
    centrality = np.zeros(n)
    for j in range(1, n):
        # Sum over all other nodes i of demand_i / distance(i,j)
        # Exclude j itself to avoid self-division
        dists = distance_matrix[:, j].copy()
        dists[j] = eps  # Avoid self-division
        centrality[j] = np.sum(demand_weights / dists)
    
    # Normalize centrality to [0, 1] range
    c_min = centrality[1:].min() if n > 1 else 0
    c_max = centrality[1:].max() if n > 1 else 1
    if c_max > c_min:
        centrality_norm = (centrality - c_min) / (c_max - c_min)
    else:
        centrality_norm = np.ones(n)
    centrality_norm[0] = 1.0  # Depot gets neutral centrality
    
    # Compute spatial density: for each node, how much demand is nearby
    # Use a Gaussian kernel with bandwidth based on the problem scale
    coords = np.asarray(coordinates, dtype=np.float64)
    # Estimate bandwidth as mean nearest-neighbor distance
    if n > 1:
        # Compute pairwise distances
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(pairwise_dists, np.inf)
        nn_dists = np.min(pairwise_dists, axis=1)
        bandwidth = np.mean(nn_dists[1:]) * 2.0  # Use 2x mean NN distance
    else:
        bandwidth = 1.0
    
    # Compute spatial density for each node
    density = np.zeros(n)
    for j in range(1, n):
        dists_to_j = np.sqrt(np.sum((coords - coords[j]) ** 2, axis=1))
        # Gaussian kernel
        kernel = np.exp(-0.5 * (dists_to_j / bandwidth) ** 2)
        density[j] = np.sum(demand_weights * kernel)
    
    # Normalize density
    d_min = density[1:].min() if n > 1 else 0
    d_max = density[1:].max() if n > 1 else 1
    if d_max > d_min:
        density_norm = (density - d_min) / (d_max - d_min)
    else:
        density_norm = np.ones(n)
    density_norm[0] = 1.0
    
    # Combine: heuristic[i,j] = inv_dist[i,j] * (0.5 * centrality_norm[j] + 0.5 * density_norm[j])
    # The target node's centrality and density boost the edge attractiveness
    target_score = 0.5 * centrality_norm + 0.5 * density_norm
    target_score[0] = 1.0  # Depot is neutral
    
    # Build the heuristic matrix: each column j is scaled by target_score[j]
    heuristic = inv_dist * target_score[np.newaxis, :]
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic

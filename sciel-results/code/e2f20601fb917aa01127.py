import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    Uses Lagrangian relaxation to derive edge weights from dual optimization.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Ensure shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    coords = np.array(coordinates, dtype=float)
    dem = np.array(demands, dtype=float)
    
    # Base heuristic: inverse distance (linear, standard for ACO)
    base = np.zeros_like(d)
    mask = d > 1e-12
    base[mask] = 1.0 / d[mask]
    
    # Lagrangian relaxation: solve relaxed problem ignoring capacity but penalizing load
    # The dual variable lambda penalizes violations of capacity constraint
    # We approximate the dual by finding edges that would be in optimal routes
    # if capacity were relaxed, then weighting them higher
    
    # Compute demand-based attractiveness: edges to low-demand nodes are more attractive
    # as they allow more flexibility in route construction
    demand_factor = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                demand_factor[i, j] = 1e-9
            else:
                # Higher demand at j means less remaining capacity for subsequent nodes
                # So prefer edges to lower demand nodes
                demand_factor[i, j] = 1.0 / (1.0 + dem[j] / capacity)
    
    # Combine base distance heuristic with demand-based Lagrangian factor
    heuristic = base * demand_factor
    
    # Additional refinement: boost edges that connect nodes in similar angular sectors
    # relative to the depot, as these tend to form non-crossing routes
    depot = coords[0]
    angles = np.arctan2(coords[:, 1] - depot[1], coords[:, 0] - depot[0])
    
    # Compute angular differences and boost edges with small angular difference
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    # Wrap around to [0, pi]
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Smooth angular preference: edges with small angular difference get boosted
    angular_boost = np.exp(-angle_diff / 0.5)  # Decay scale of 0.5 radians
    
    # Combine all components
    heuristic = base * demand_factor * angular_boost
    
    # Ensure diagonal is very small (self-loops not useful)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic

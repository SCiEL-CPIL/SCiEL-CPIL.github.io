"""Optimized heuristics for ACO CVRP using rank-based distance with depot proximity boost."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    - Use rank-based distance discrimination: instead of raw inverse distance,
      use the ordinal rank of each edge's distance among all edges from that node.
      This creates a smoother, more uniform gradient that prevents the nearest
      neighbor from dominating the probability landscape.
    - Add a depot-proximity boost: edges to/from the depot (node 0) get an extra
      multiplicative factor based on how close the customer is to the depot,
      encouraging routes that start/end with nearby customers.
    - Normalize so mean off-diagonal value is ~1.0 to balance with initial
      pheromone of 1.0.
    """
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)

    eps = 1e-9

    # Set diagonal to infinity for ranking purposes (we don't want self-edges)
    d_rank = d.copy()
    np.fill_diagonal(d_rank, np.inf)

    # Compute rank-based heuristic: for each row, rank the distances
    # Lower distance = better = higher rank value
    # Use argsort to get ranks (0-based), then invert so closest gets highest value
    eta = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Get sorted indices by distance for row i
        sorted_idx = np.argsort(d_rank[i])
        # Assign rank values: closest gets n-1, farthest gets 0
        # Then scale to create a smooth gradient
        ranks = np.zeros(n, dtype=np.float64)
        for rank_pos, idx in enumerate(sorted_idx):
            # Use a power-law scaling: rank^0.7 gives moderate discrimination
            ranks[idx] = (n - 1 - rank_pos) ** 0.7
        eta[i, :] = ranks

    # Depot proximity boost:
    # For edges involving the depot (node 0), boost based on customer's distance to depot
    if n > 1:
        # Distance of each customer to depot
        depot_dist = d[0, 1:].copy()
        # Normalize: closer customers get higher boost
        # Boost factor: 1 + alpha * (1 - normalized_distance)
        # where normalized_distance is in [0, 1]
        max_dd = np.max(depot_dist) if np.max(depot_dist) > 0 else 1.0
        min_dd = np.min(depot_dist) if np.min(depot_dist) > 0 else 0.0
        range_dd = max_dd - min_dd if max_dd > min_dd else 1.0
        norm_dd = (depot_dist - min_dd) / range_dd  # 0 = closest, 1 = farthest
        
        alpha = 0.3  # Maximum 30% boost for closest-to-depot customers
        
        # Boost edges from depot to customers
        boost = 1.0 + alpha * (1.0 - norm_dd)
        eta[0, 1:] *= boost
        
        # Boost edges from customers to depot
        eta[1:, 0] *= boost

    # Normalize: mean off-diagonal should be ~1.0
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])

    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        # Fallback: simple inverse distance
        eta = 1.0 / ((d + eps) ** 1.0)

    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9

    return eta
# EVOLVE-BLOCK-END

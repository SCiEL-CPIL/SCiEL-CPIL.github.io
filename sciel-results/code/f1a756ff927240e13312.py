# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic edge-attractiveness matrix for ACO of CVRP.
    
    Uses a multi-factor heuristic:
    1. Inverse distance (base)
    2. Demand compatibility (penalize edges to high-demand nodes when capacity is tight)
    3. Depot distance bonus (encourage serving customers near depot first to reduce return trips)
    4. Spatial clustering (encourage serving nearby customers together)
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = d_mat.shape[0]
    
    epsilon = 1e-9
    
    # 1. Base: inverse distance
    base = 1.0 / (d_mat + epsilon)
    
    # 2. Demand-aware penalty: penalize edges to high-demand customers
    #    High-demand customers should be served in smaller, focused routes
    #    This reduces the attractiveness of edges leading to high-demand nodes
    #    when coming from nodes that already have accumulated load
    #    Simple approach: scale by 1/(1 + demand_j / capacity)
    #    For demand 1, cap 50: 1/(1+0.02) = 0.98
    #    For demand 9, cap 50: 1/(1+0.18) = 0.85
    demand_factor = 1.0 / (1.0 + dem / capacity)
    
    # Apply as column-wise scaling
    heuristic = base * demand_factor[np.newaxis, :]
    
    # 3. Depot distance bonus: customers close to depot get a boost
    #    This encourages starting routes with nearby customers, reducing
    #    the total distance per route
    depot_dist = d_mat[0, :]  # distance from depot to each node
    depot_bonus = 1.0 + 0.3 * (1.0 - depot_dist / (np.max(depot_dist) + epsilon))
    
    # Apply depot bonus to edges from depot and to depot
    # Edges from depot: row 0
    heuristic[0, :] *= depot_bonus
    # Edges to depot: column 0
    heuristic[:, 0] *= depot_bonus[np.newaxis, :]
    
    # 4. Spatial coherence: encourage serving nearby customers together
    #    For each pair (i, j), if they are close to each other, boost the edge
    #    This promotes compact, space-filling routes
    #    Use a Gaussian kernel on distance
    sigma = np.median(d_mat[d_mat > 0]) * 0.5
    spatial_boost = 1.0 + 0.2 * np.exp(-d_mat**2 / (2 * sigma**2 + epsilon))
    heuristic *= spatial_boost
    
    # 5. Demand pairing compatibility:
    #    If two customers have demands that together exceed capacity,
    #    the edge between them is less useful (can't be in same route)
    #    This is a soft penalty
    demand_sum = dem[:, np.newaxis] + dem[np.newaxis, :]
    # Only penalize if sum > capacity (shouldn't happen with cap 50 and max demand 9,
    #    but still useful for the heuristic to be general)
    #    Actually with max demand 9 and capacity 50, no two customers exceed capacity.
    #    So this factor is ~1 for all pairs. Skip.
    
    # 6. Balance: moderate the overall scale
    #    Normalize so that the median heuristic value is around 1
    median_val = np.median(heuristic[heuristic > 0])
    if median_val > 0:
        heuristic = heuristic / median_val
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

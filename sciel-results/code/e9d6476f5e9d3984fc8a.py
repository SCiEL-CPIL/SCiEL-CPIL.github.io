"""
Node-potential field heuristic for ACO CVRP.
Replaces pairwise edge attraction with a spatial density and load-balancing pressure model.
The heuristic for edge (i, j) is:
  eta[i, j] = (1 / dist[i, j]) * exp(-alpha * |coord[j] - centroid_remaining|) * (1 + beta * demand[j])
This encourages ants to visit customers in dense regions with high demand, promoting compact routes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a node-potential field approach.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    # Validate inputs
    if distance_matrix.shape != (n, n):
        return 1.0 / (distance_matrix + 1e-9)

    # Precompute
    coords = np.asarray(coordinates)
    demands = np.asarray(demands, dtype=float)
    
    # Base inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Compute centroid of all customers (excluding depot)
    # In a static heuristic, we approximate the "remaining" centroid as the global centroid
    # or use a weighted centroid based on demand.
    # A more dynamic approach would be hard to precompute for a static matrix, 
    # so we use a spatial potential field.
    
    # Strategy: 
    # 1. Compute a "density" or "potential" for each node based on its neighbors.
    # 2. The potential represents how "attractive" it is to visit this node given its location.
    # 3. Combine with demand to prioritize high-demand nodes in dense areas.
    
    # Let's try a different structural change: 
    # Use a "space-filling" or "cluster" based heuristic.
    # Nodes that are closer to the centroid of the demand cloud are more attractive.
    
    # Compute demand-weighted centroid of customers
    customer_indices = np.arange(1, n)
    total_demand = demands[customer_indices].sum()
    if total_demand > 0:
        centroid = np.average(coords[customer_indices], axis=0, weights=demands[customer_indices])
    else:
        centroid = coords[customer_indices].mean(axis=0)
        
    # Distance of each node to the centroid
    dist_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    # Normalize distances for exponential decay
    max_dist = dist_to_centroid.max()
    if max_dist == 0:
        max_dist = 1.0
        
    # Potential: Exponential decay from centroid. 
    # Nodes closer to centroid are more attractive.
    # We can also add a local density term.
    
    # Local density: For each node, sum of inverse distances to its k-nearest neighbors
    # This creates a "field" where dense areas have higher potential.
    
    k = 5
    local_density = np.zeros(n)
    for i in range(1, n):
        if i == 0:
            continue
        dists = distance_matrix[i, 1:].copy()
        # Exclude self
        dists[i-1] = np.inf
        nearest_dists = np.sort(dists)[:k]
        # Inverse sum, clipped
        local_density[i] = np.sum(1.0 / (nearest_dists + 1e-9))
        
    # Normalize local density
    max_density = local_density.max()
    if max_density > 0:
        local_density = local_density / max_density
        
    # Combine potential terms:
    # 1. Centroid attraction: exp(-alpha * dist_to_centroid / max_dist)
    # 2. Local density: local_density
    # 3. Demand factor: (1 + beta * demand_ratio)
    
    # Parameters to tune
    alpha = 1.5  # Strength of centroid attraction
    beta = 0.5   # Strength of demand preference
    
    demand_ratio = demands / max(demands.max(), 1e-9)
    
    # Node potential vector
    node_potential = np.ones(n)
    node_potential[0] = 1.0 # Depot is neutral
    
    for i in range(1, n):
        centroid_term = np.exp(-alpha * (dist_to_centroid[i] / max_dist))
        density_term = local_density[i]
        demand_term = 1.0 + beta * demand_ratio[i]
        
        # Multiplicative combination
        node_potential[i] = centroid_term * (0.5 + 0.5 * density_term) * demand_term
        
    # Construct edge heuristic matrix
    # eta[i, j] = inv_dist[i, j] * node_potential[j]
    # This makes the attractiveness of moving to j depend on j's potential.
    
    heuristic_matrix = inv_dist * node_potential[np.newaxis, :]
    
    # Ensure positive
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    # Optional: Quick validation simulation to ensure it's not worse than base
    # If time permits, we can run a quick greedy check.
    if time.time() < deadline - 5:
        # Simple sanity check: does it produce a valid tour?
        # We trust the math, but let's ensure no NaNs.
        if not np.all(np.isfinite(heuristic_matrix)):
            heuristic_matrix = inv_dist.copy()
            
    return heuristic_matrix
# EVOLVE-BLOCK-END

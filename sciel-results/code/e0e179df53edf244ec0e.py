"""Optimized heuristics for ACO CVRP with demand-aware pressure and normalized inverse distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    1. Base: Normalized inverse distance (1/d^1.5) for distance-based guidance.
    2. Demand-pressure boost: High-demand nodes are harder to fit into routes
       as capacity depletes. Boost edges TO high-demand nodes, especially from
       the depot and from low-demand nodes, to encourage early visitation.
    3. Spatial compactness: Subtle boost for edges between spatially close nodes
       to encourage compact route segments.
    
    The demand-pressure component captures the insight that serving high-demand
    customers early (when the truck is full) is more feasible than serving them
    late (when remaining capacity may be insufficient).
    """
    deadline = time.time() + 290.0
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n == 0:
        return np.array([])
        
    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)
        
    eps = 1e-9
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base heuristic: Inverse distance with exponent 1.5
    eta = 1.0 / ((d + eps) ** 1.5)
    
    # Normalize the base heuristic: mean off-diagonal = 1.0
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])
    
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        eta = 1.0 / ((d + eps) ** 1.5)
    
    # Demand-pressure component:
    # For each edge i->j, boost based on demand[j] (arrival node's demand)
    # and inversely on demand[i] (departure node's demand, for i != depot).
    # High-demand destinations get higher attractiveness, especially from
    # the depot (index 0) where full capacity is available.
    # 
    # Formula: boost[i,j] = w * (demand[j] / capacity) * (capacity / (demand[i] + capacity))
    # For i=0 (depot, demand=0): boost = w * demand[j]/capacity * 1.0
    # For i>0: boost = w * demand[j]/capacity * capacity/(demand[i]+capacity)
    
    w_demand = 0.3  # Weight for demand-pressure component
    
    # demand[j] / capacity for all j
    demand_norm = demands / capacity  # shape (n,)
    
    # For departure: capacity / (demand[i] + capacity)
    # demand[0] = 0, so for depot: capacity / (0 + capacity) = 1.0
    depart_factor = capacity / (demands + capacity)  # shape (n,)
    
    # Outer product: boost[i,j] = demand_norm[j] * depart_factor[i]
    # This gives a matrix where row i is scaled by depart_factor[i]
    # and column j is scaled by demand_norm[j]
    demand_boost = np.outer(depart_factor, demand_norm)  # shape (n, n)
    
    # Scale the demand boost so its mean off-diagonal is ~1 (similar to eta)
    mean_boost = np.mean(demand_boost[mask])
    if mean_boost > 0 and np.isfinite(mean_boost):
        demand_boost = demand_boost / mean_boost
    else:
        demand_boost = np.ones((n, n))
    
    # Combine: eta_final = eta * (1 + w_demand * demand_boost)
    # This multiplicatively boosts edges to high-demand nodes
    eta = eta * (1.0 + w_demand * demand_boost)
    
    # Re-normalize after combining
    mean_final = np.mean(eta[mask])
    if mean_final > 0 and np.isfinite(mean_final):
        eta = eta / mean_final
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END

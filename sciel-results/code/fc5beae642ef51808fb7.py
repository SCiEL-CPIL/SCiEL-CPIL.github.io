"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for CVRP using a combination of
    distance-based and demand-based heuristics.
    
    The heuristic combines:
    1. Inverse distance (primary signal)
    2. A penalty for edges that would lead to poor load utilization
    3. A bonus for edges connecting nearby customers with compatible demands
    
    Larger values = more attractive edges.
    """
    n = distance_matrix.shape[0]
    
    # Ensure demands are treated as floats
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    base = 1.0 / dist_safe
    
    # Demand compatibility factor:
    # For edge (i, j), if we're at node i and considering going to j,
    # we want j's demand to be "reasonable" relative to remaining capacity.
    # Since we don't know remaining capacity at heuristic computation time,
    # we use a static heuristic: prefer edges where the destination demand
    # is moderate (not too small, not too large relative to capacity).
    
    # Create a demand score matrix: for each (i, j), score based on demand[j]
    # We want to avoid very small demands (inefficient) and very large demands (risky)
    # Ideal demand is around capacity/3 to capacity/2 for good vehicle utilization
    
    # Demand attractiveness: peaks around capacity/3
    ideal_demand = capacity / 3.0
    demand_deviation = np.abs(demands[1:] - ideal_demand)  # for customers 1..n-1
    # Normalize and create a smooth preference
    max_dev = max(ideal_demand, 1.0)
    demand_score = np.exp(-0.5 * (demand_deviation / max_dev) ** 2)
    
    # Build full demand score matrix (depot row/col get neutral score)
    demand_matrix = np.ones((n, n), dtype=np.float64)
    demand_matrix[:, 1:] = demand_score[None, :]  # column j gets score based on demand[j]
    demand_matrix[0, :] = 1.0  # from depot, demand score is neutral
    demand_matrix[:, 0] = 1.0  # to depot, demand score is neutral
    demand_matrix[0, 0] = 1.0
    
    # Combine: base * (0.7 + 0.3 * demand_score)
    # This gives 70% weight to distance, 30% to demand compatibility
    combined = base * (0.7 + 0.3 * demand_matrix)
    
    # Additional refinement: penalize long edges more heavily for far nodes
    # Use a distance exponent > 1 to make long distances less attractive
    # This creates a stronger preference for local moves
    dist_exponent = 1.5
    dist_exp = np.power(dist_safe, dist_exponent)
    base_exp = 1.0 / dist_exp
    
    # Blend the two distance-based heuristics
    blend = 0.5 * base + 0.5 * base_exp * (1.0 / np.mean(dist_exp))  # normalize
    
    # Actually, let's be more careful. Let's use:
    # heuristic = (1/d^1.2) * demand_factor
    # This is simpler and more interpretable
    
    final_dist = np.power(dist_safe, 1.2)
    heuristic = (1.0 / final_dist) * (0.7 + 0.3 * demand_matrix)
    
    # Ensure no zeros or negatives
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Normalize to have reasonable magnitude (mean around 1)
    mean_val = np.mean(heuristic)
    if mean_val > 0:
        heuristic = heuristic / mean_val
    
    return heuristic
# EVOLVE-BLOCK-END

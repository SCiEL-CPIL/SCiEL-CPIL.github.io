"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    The heuristic combines:
    1. Inverse distance (primary factor)
    2. Demand-based penalty: edges leading to customers with high demand relative to 
       remaining capacity are less attractive
    3. A small bonus for edges that keep the route balanced
    
    Returns a matrix of the same shape as distance_matrix where larger values 
    indicate more attractive edges.
    """
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-aware adjustment:
    # For each edge i->j, if j is a customer (j > 0), consider how "expensive" 
    # it is in terms of demand relative to capacity
    # We want to prefer edges to customers with moderate demand that fit well
    
    # Create a demand ratio matrix: for each destination j, demand[j]/capacity
    # This penalizes going to high-demand customers slightly
    demand_ratio = np.zeros(n)
    demand_ratio[1:] = np.asarray(demands[1:], dtype=float) / capacity
    
    # For each edge i->j, apply a mild penalty based on demand_ratio[j]
    # This makes edges to high-demand customers slightly less attractive
    # But we keep the effect modest so distance still dominates
    
    # Shape: (n, n) where each column j is scaled by demand_ratio[j]
    # We use a soft penalty: multiply by (1 - alpha * demand_ratio[j])
    # where alpha is small (e.g., 0.3) to keep distance as primary factor
    alpha = 0.3
    penalty = 1.0 - alpha * demand_ratio[np.newaxis, :]
    
    # Ensure penalty is positive
    penalty = np.maximum(penalty, 0.1)
    
    # Apply penalty to inverse distance
    heuristic = inv_dist * penalty
    
    # Set depot edges (row 0) to be based purely on inverse distance
    # since the depot has no demand constraint
    heuristic[0, :] = inv_dist[0, :]
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

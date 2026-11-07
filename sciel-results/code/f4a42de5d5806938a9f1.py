import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a demand-aware heuristic matrix.
    
    This approach introduces a soft penalty for high-demand customers.
    The heuristic is defined as:
    H_ij = 1 / ( (d(i, j) + epsilon) * sqrt(1 + demand_j / capacity) )
    
    This encourages ants to prefer shorter distances, while gently discouraging
    immediate moves to customers that consume a large portion of the remaining capacity,
    potentially allowing for better packing of multiple customers per trip.
    """
    eps = 1e-9
    
    # Ensure demands are float
    demands = np.asarray(demands, dtype=float)
    
    # Calculate demand factor: sqrt(1 + demand/capacity)
    # Using square root to soften the penalty compared to linear scaling
    demand_factor = np.sqrt(1.0 + demands / capacity)
    
    # Distance term
    distance_term = distance_matrix + eps
    
    # Combine: Inverse of (Distance * Demand Factor)
    # Broadcasting demand_factor (N,) across columns (j) of distance_term (N, N)
    heuristic = 1.0 / (distance_term * demand_factor)
    
    # Ensure no non-finite values and positivity
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic

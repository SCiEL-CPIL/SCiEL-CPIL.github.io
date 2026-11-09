"""Optimized heuristics for ACO CVRP combining inverse distance with demand-aware weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Combines inverse distance with a mild demand-based penalty to guide
    ants toward feasible, efficient routes.
    
    Strategy:
    - Base: inverse distance (larger for closer nodes)
    - Refinement: slightly penalize edges to high-demand customers when
      the current node is also high-demand, to encourage load balancing.
    - Use a softened exponent to avoid over-concentration.
    """
    deadline = time.time() + 290.0  # Safety margin before 300s limit
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Validate shape
    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)
    
    # Base heuristic: inverse distance with a mild power to flatten extremes
    # Using power 0.8 instead of 1.0 to reduce dominance of very short edges
    eps = 1e-9
    base = 1.0 / np.power(d + eps, 0.8)
    
    # Demand-aware adjustment:
    # For each edge (i, j), if both i and j have high demand, slightly reduce
    # attractiveness to encourage spreading out high-demand stops.
    # This is a soft penalty: multiply by (1 - small_factor * demand_product_normalized)
    dem = np.asarray(demands, dtype=np.float64)
    max_dem = np.max(dem) if np.max(dem) > 0 else 1.0
    norm_dem = dem / max_dem  # Normalize to [0, 1]
    
    # Outer product of normalized demands: high where both endpoints are high-demand
    dem_prod = np.outer(norm_dem, norm_dem)
    
    # Small penalty factor: 0.1 means at most 10% reduction for highest-demand pairs
    penalty_factor = 0.1
    adjustment = 1.0 - penalty_factor * dem_prod
    
    # Combine
    eta = base * adjustment
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END

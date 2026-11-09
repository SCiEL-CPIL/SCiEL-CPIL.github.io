import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    d = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    # Base distance heuristic with power 4.5
    heuristic = np.zeros_like(d)
    mask = d > 0
    heuristic[mask] = 1.0 / (d[mask] ** 4.5)
    
    # Angular coherence: reward edges where both nodes are in similar angular sectors
    # from the depot. This encourages ants to build routes that sweep around the
    # depot rather than zigzagging across it.
    depot = coords[0]
    angles = np.arctan2(coords[1:, 1] - depot[1], coords[1:, 0] - depot[0])
    # Convert to [0, 2*pi) for smoother angular distance
    angles = angles % (2 * np.pi)
    
    # Angular distance between each pair of customers (i, j) for i,j >= 1
    # Use circular distance: min(|a-b|, 2*pi-|a-b|)
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Smooth coherence score: higher when angular difference is small
    # Using a soft exponential decay with sigma ~ 0.5 radians
    sigma = 0.5
    coherence = np.exp(-angle_diff / sigma)
    
    # Build full n x n coherence matrix (depot row/col gets neutral 1.0)
    full_coherence = np.ones((n, n))
    full_coherence[1:, 1:] = coherence
    
    # Weighted blend: 70% distance, 30% coherence
    heuristic = 0.7 * heuristic + 0.3 * heuristic * full_coherence
    
    # Demand-aware boost: favor destinations with higher demand
    demand_boost = 1.0 + 0.08 * (demands / capacity)
    heuristic *= demand_boost[np.newaxis, :]
    
    # Ensure all values are positive (non-positive becomes 1e-9 per contract)
    heuristic[heuristic <= 0] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic

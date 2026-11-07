import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of CVRP.
    
    Structural Change: Uses a Savings-Heuristic Inspired static matrix.
    Instead of generic compatibility, we boost edges that represent high
    "savings" relative to serving nodes individually from the depot.
    This aligns with the fundamental logic of constructing efficient routes.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    
    # Base heuristic: Inverse distance
    eps = 1e-9
    base = 1.0 / (d + eps)
    
    # Calculate Savings for each pair (i, j)
    # Saving = dist(i, 0) + dist(0, j) - dist(i, j)
    # We normalize this to be between 0 and 1 (or similar)
    # d_0 is the vector of distances from depot (node 0) to all nodes
    d_0 = d[0, :]
    
    # Savings matrix S[i, j] = d_0[i] + d_0[j] - d[i, j]
    # Note: d is symmetric, so d[i, j] == d[j, i]
    S = d_0[:, np.newaxis] + d_0[np.newaxis, :] - d
    
    # Normalize savings to [0, 1] range for stable multiplication
    # Max possible saving is roughly max(d_0) * 2
    max_saving = np.max(S)
    if max_saving > 0:
        S_norm = S / max_saving
    else:
        S_norm = np.zeros_like(S)
        
    # Clamp negative savings to 0 (triangle inequality usually prevents this, but floating point might not)
    S_norm = np.clip(S_norm, 0, 1)
    
    # Demand-awareness:
    # High demand nodes are "costly" to include. 
    # We can slightly down-weight edges to high demand nodes if they are far,
    # but the savings term already handles the geometry well.
    # Let's add a mild demand penalty to the base distance component.
    # If a node has high demand, it consumes more capacity, potentially forcing
    # an earlier return. This is a second-order effect.
    # Let's stick to the Savings structure as the primary driver.
    
    # Combine: Heuristic = Base * (1 + alpha * S_norm)
    # alpha controls how much we trust the savings signal over pure distance.
    alpha = 1.5
    
    heuristic = base * (1.0 + alpha * S_norm)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Handle non-finite and non-positive values
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic

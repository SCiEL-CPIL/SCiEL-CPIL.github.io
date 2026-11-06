"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n = adjacency.shape[0]
    
    # Calculate degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of neighbor degrees
    neighbor_deg_sum = adjacency @ deg
    
    # Calculate mean neighbor degree (avoid division by zero)
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Strategy: Strongly prefer low degree, use MND as secondary signal
    # Scale degree to dominate, MND as tiebreaker with meaningful weight
    # Normalize MND to be in similar range as degree for better balance
    # For ER(200, p), deg ~ N(0, p*199), MND ~ similar range
    # Use -deg * 10 + MND to make degree primary, MND secondary
    
    scores = -deg * 10.0 + mean_neighbor_deg
    
    # Check deadline
    if time.time() > deadline:
        return -deg
    
    return scores
# EVOLVE-BLOCK-END

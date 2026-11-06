import numpy as np
import time

def simulate(scores, adjacency):
    n = len(scores)
    order = np.argsort(scores)[::-1]
    selected = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[adjacency[v]].any():
            selected[v] = True
            count += 1
    return count

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # Initial scores: prefer low degree, break ties with high mean neighbor degree
    # This is a strong baseline for MIS
    safe_deg = deg + 1.0
    inv_deg = 1.0 / safe_deg
    neighbor_sum_inv_deg = adjacency @ inv_deg
    neighbor_deg_sum = adjacency @ deg
    mean_neighbor_deg = np.zeros(n)
    mask = deg > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / deg[mask]
    
    # Weighting: Degree is primary, neighbor degree is secondary
    # We want low degree, so -deg.
    # We want high mean neighbor degree, so +mean_neighbor_deg.
    # Scale mean_neighbor_deg so it doesn't dominate degree if degrees vary widely.
    # In ER(200, p), degrees are concentrated.
    scores = -deg + 0.1 * mean_neighbor_deg
    
    current_size = simulate(scores, adjacency)
    
    # Local Search: Swap pairs of vertices in the score list
    # This effectively reorders the greedy selection
    while time.time() < deadline:
        # Pick two random distinct indices
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        new_size = simulate(scores, adjacency)
        
        if new_size > current_size:
            current_size = new_size
            # Accept swap
        else:
            # Revert swap
            scores[i], scores[j] = scores[j], scores[i]
            
    return scores

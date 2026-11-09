import numpy as np
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    deadline = time.time() + 110  # safety margin
    
    # Initial score: inverse degree (low degree = high priority)
    degrees = adjacency.sum(axis=1).astype(float)
    degrees[degrees == 0] = 1
    scores = 1.0 / degrees
    
    # Iterate to smooth: score(v) should account for neighbor scores
    # If neighbors have high scores, picking v is costly -> lower score
    # score_new(v) = base(v) - lambda * mean(score(neighbors(v)))
    # But we want this to converge to a stable ordering
    
    # Use core decomposition as a strong signal
    # Compute core numbers
    adj_bool = adjacency.astype(bool)
    alive = np.ones(n, dtype=bool)
    current_deg = adjacency.sum(axis=1).astype(int).copy()
    core = np.zeros(n, dtype=int)
    
    # Simple O(n^2) core decomposition for n=200
    for _ in range(n):
        if not alive.any():
            break
        # Find vertex with minimum current degree among alive
        min_deg = np.inf
        min_v = -1
        for v in range(n):
            if alive[v] and current_deg[v] < min_deg:
                min_deg = current_deg[v]
                min_v = v
        if min_v < 0:
            break
        core[min_v] = min_deg
        alive[min_v] = False
        # Update degrees of neighbors
        neighbors = np.where(adj_bool[min_v])[0]
        for nb in neighbors:
            if alive[nb]:
                current_deg[nb] -= 1
    
    # Core number: lower core = peripheral = should be picked first (higher priority)
    # So priority = -core number (higher = picked first)
    # But also incorporate degree: among same core, lower degree is better
    
    # Combine: score = -core * 1000 + (1 / (degree + 1))
    # This makes core the dominant factor
    
    # But let's try iterative refinement
    # Start with core-based scores
    scores = -core.astype(float) * 100.0
    
    # Add a small degree-based component
    scores += 1.0 / (degrees + 1.0)
    
    # Iterate a few times to smooth
    for iteration in range(50):
        if time.time() > deadline:
            break
        # For each vertex, compute mean score of neighbors
        neighbor_sum = adjacency.astype(float) @ scores
        neighbor_count = degrees
        neighbor_count[neighbor_count == 0] = 1
        mean_neighbor_score = neighbor_sum / neighbor_count
        
        # New score: base (core) - penalty for having high-scoring neighbors
        # If neighbors have high scores (picked early), picking me wastes them
        new_scores = -core.astype(float) * 100.0 - 0.1 * mean_neighbor_score
        new_scores += 1.0 / (degrees + 1.0)
        
        # Check convergence
        if np.allclose(scores, new_scores, atol=1e-6):
            scores = new_scores
            break
        scores = new_scores
    
    return scores

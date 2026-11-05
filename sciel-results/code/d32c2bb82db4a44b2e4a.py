"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    We use a structural heuristic based on the "cost" of selecting a vertex.
    Selecting a vertex v removes it and all its neighbors from the candidate pool.
    The "cost" is higher if the neighbors are high-degree (hubs), because removing a hub
    eliminates many other potential connections/vertices.
    
    We define the cost of vertex v as C_v = sum(deg(u) for u in N(v)) + deg(v).
    This combines the sum of neighbor degrees (SND) with the vertex's own degree.
    - SND penalizes vertices connected to hubs.
    - Own degree penalizes high-degree vertices directly (tie-breaking).
    
    We want to pick vertices with LOW cost, so we set score = -C_v.
    This effectively implements a "Minimum Sum-of-Neighbor-Degrees" strategy with a degree tie-breaker,
    which is known to perform well on sparse graphs like Erdos-Renyi by preferring peripheral
    nodes (leaves) and avoiding central hubs.
    """
    start_time = time.time()
    
    # Ensure float matrix for efficient operations, though int is fine for sum
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Compute degrees
    deg = A.sum(axis=1)
    
    # 2. Compute Sum of Neighbor Degrees (SND)
    # s_v = sum_{u in N(v)} deg(u)
    # This is equivalent to A @ deg
    snd = A @ deg
    
    # 3. Calculate Score
    # We want to minimize (snd + deg).
    # Using a linear combination ensures that if snd values are close,
    # lower degree vertices are preferred.
    # The weight of 1.0 for both is a simple, robust choice.
    # If snd is very large, it dominates, which is desirable.
    cost = snd + deg
    score = -cost
    
    # Safety check for time (though this is very fast)
    if time.time() - start_time > 10.0:
        # Fallback to simple degree-based score if something went wrong
        return -deg.astype(float)
        
    return score
# EVOLVE-BLOCK-END

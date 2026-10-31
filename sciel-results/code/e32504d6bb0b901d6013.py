import numpy as np
import itertools
import time

D = 11

def generate_weight_5_vectors():
    """Generate all vectors of weight 5 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy MIS with random tie-breaking.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Generate candidates
    vectors = generate_weight_5_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    # Vectors are int8, dot products fit in int16 or int32
    dots = vectors @ vectors.T
    # Conflict if dot > 2
    conflicts = dots > 2
    
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    # The greedy is fast, so we can run many restarts
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different random seeds for different restarts
        current_rng = np.random.default_rng(42 + i)
        current_set = greedy_mis(conflicts, deadline, current_rng)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)

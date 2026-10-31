import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline):
    """
    Greedy Maximum Independent Set on a graph defined by boolean conflict matrix.
    conflicts[i, j] is True if i and j cannot be in the same set.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    # Initial degrees
    degrees = conflicts.sum(axis=1)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        # degrees is updated dynamically
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[np.random.randint(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        # For each v in alive, degree[v] decreases by number of neighbors in to_remove
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            # Count conflicts between remaining_alive and to_remove
            # conflicts[remaining_alive, to_remove] is (len(rem), len(to_remove))
            # Sum along axis 1 gives count of removed neighbors for each remaining node
            # Note: conflicts is symmetric, so this works.
            # We must ensure we don't count self if chosen is in to_remove (it is).
            # But chosen is not in remaining_alive.
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Generate candidates
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    # vectors shape (N, D)
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2
    # Note: dot[i,i] = 4 > 2, so self is a conflict.
    # We want to select distinct vectors, so self-conflict is fine (we never pick self twice).
    # The MIS algorithm handles this by removing 'chosen' from 'alive'.
    conflicts = dots > 2
    
    # Run Greedy MIS
    selected_indices = greedy_mis(conflicts, deadline)
    
    if not selected_indices:
        # Fallback: return empty or a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[selected_indices]
    
    # Optional: Verify and clean up if time permits?
    # The greedy ensures dot <= 2 for all selected pairs.
    
    return result_vectors.astype(np.int64)

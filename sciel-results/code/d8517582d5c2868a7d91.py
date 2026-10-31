import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate all weight-4 vectors
    # Indices 0..10
    # Choose 4 indices
    # Assign signs +/-
    pool = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Precompute Gram matrix G
    # G[i, j] = pool[i] . pool[j]
    # We need G[i, j] <= 2 for compatibility
    G = pool @ pool.T
    
    # Precompute degrees (number of conflicts)
    # Conflict if G[i, j] > 2
    # Diagonal is 4, which is > 2, but we don't select same vector twice.
    # We can ignore diagonal or handle it.
    degrees = np.sum(G > 2, axis=1)
    
    best_set = []
    best_len = 0
    best_indices = []
    
    # Strategy 1: Random restarts
    # We can run many of these.
    # Each restart is O(K * N) with small constant.
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        # Random permutation
        order = np.random.permutation(n_pool)
        
        selected_indices = []
        # Mask of compatible vectors
        # Start with all True
        # We can just keep a list of selected and check G[idx, selected] <= 2
        # But updating a mask is faster.
        
        # Let's use the mask approach.
        # mask[i] is True if pool[i] is compatible with all currently selected.
        # Initially, all True.
        # When we select k, we do mask = mask & (G[k, :] <= 2)
        # Then we pick the first True index in 'order'.
        
        # However, 'order' is a permutation.
        # We need to iterate through 'order' and pick the first one where mask is True.
        # Then update mask.
        
        # Optimization:
        # Instead of checking mask for every item in order,
        # we can just iterate order and check.
        
        # But updating mask is O(N).
        # If we do this K times, it's O(K*N).
        # N=5280, K~500. 2.6M ops. Fast.
        
        mask = np.ones(n_pool, dtype=bool)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
                
            if mask[idx]:
                # Select this vector
                selected_indices.append(idx)
                # Update mask: intersect with compatibility of idx
                # G[idx, :] <= 2
                # Note: G[idx, idx] = 4 > 2, so mask[idx] becomes False.
                mask &= (G[idx, :] <= 2)
                
        current_len = len(selected_indices)
        if current_len > best_len:
            best_len = current_len
            best_indices = selected_indices
            # Optional: if we hit 504, we can stop early?
            # 504 is the known bound. If we hit it, we can't do better with this method?
            # Actually, we might find 504.
            if best_len >= 504:
                break

    # If we still have time, try heuristic restarts (sorted by degree)
    if time.time() < deadline - 10:
        # Sort indices by degree ascending
        sorted_order = np.argsort(degrees)
        
        # Try a few variations of sorted order
        # 1. Pure sorted
        # 2. Sorted with random tie-breaking or noise?
        
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            
            # Add some noise to the sorted order?
            # Or just use the sorted order.
            # Since random restarts already covered random, sorted might find a different local optimum.
            
            order = sorted_order.copy()
            # Shuffle ties?
            # For simplicity, just use the sorted order.
            
            mask = np.ones(n_pool, dtype=bool)
            selected_indices = []
            
            for idx in order:
                if time.time() > deadline - 2:
                    break
                if mask[idx]:
                    selected_indices.append(idx)
                    mask &= (G[idx, :] <= 2)
            
            current_len = len(selected_indices)
            if current_len > best_len:
                best_len = current_len
                best_indices = selected_indices
                if best_len >= 504:
                    break

    if best_len == 0:
        # Fallback
        return np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return pool[best_indices]
# EVOLVE-BLOCK-END

import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Generate weight-4 vectors
    # Indices 0..10, choose 4
    # Signs +/-1
    vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int8)
    N = len(vectors)
    
    # 2. Generate weight-1 vectors (norm^2 = 4)
    # These are all compatible with weight-4 and each other
    weight1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            weight1_vectors.append(v)
    weight1_vectors = np.array(weight1_vectors, dtype=np.int64)
    
    if N == 0:
        return weight1_vectors

    # 3. Compute Conflict Matrix
    # Conflict if dot > 2
    # dots = vectors @ vectors.T
    # This is 5280 x 5280. Fast enough.
    dots = vectors @ vectors.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # 4. Greedy MIS with restarts
    # We will run as many restarts as possible before deadline
    mis_deadline = deadline - 10 # Leave 10s for final processing
    
    while time.time() < mis_deadline:
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > mis_deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            # degrees is updated only for alive nodes? 
            # We need to ensure degrees are correct for alive nodes.
            # Initially correct.
            # When we remove a node, we subtract 1 from degrees of its neighbors.
            
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
            # Only nodes that were alive and are neighbors of 'chosen' need update?
            # Actually, degrees of alive nodes that are neighbors of 'chosen' decrease by 1.
            # Neighbors of 'chosen' that are already dead don't matter.
            # So: alive_neighbors = neighbors[alive[neighbors]]
            # But 'neighbors' includes 'chosen'. 'chosen' is now dead.
            # We need to update degrees for alive nodes that are in 'neighbors'.
            
            alive_neighbors = neighbors[alive[neighbors]]
            if len(alive_neighbors) > 0:
                degrees[alive_neighbors] -= 1

        if len(selected_indices) > len(best_set):
            best_set = selected_indices
            # Optional: check time and break if we hit a theoretical max or time is tight?
            # No, keep trying.
            
    # 5. Construct final result
    if not best_set:
        return weight1_vectors
        
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors

import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # 1. Generate Norm 2 vectors (Baseline: 220 points)
    # These always form a valid set.
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution
    best_k = len(base_solution)
    
    # 2. Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph for Norm 4 vectors
    # Condition: dot product > 2 means conflict (distance < R=4)
    # We want Max Independent Set (no conflicts).
    
    # Compute full dot product matrix
    # N4 = C(11,4) * 2^4 = 330 * 16 = 5280
    # 5280x5280 matrix of int64 is ~225 MB. This is acceptable.
    # Computing it is fast.
    dots = V4 @ V4.T
    
    # Conflict if dot > 2
    # We store adjacency as a list of numpy arrays for fast boolean masking in the greedy step.
    conflict_adj = [None] * N4
    
    # Precompute neighbors
    for i in range(N4):
        if time.time() > deadline - 10:
            return best_points
        
        neighbors = np.where(dots[i] > 2)[0]
        conflict_adj[i] = neighbors

    # Free memory
    del dots

    # 4. Greedy Search with Random Restarts
    # Use numpy-based active set management for speed
    best_indices = []
    
    # Optimization: Use bitsets? 
    # N4 = 5280. Bitset of 5280 bits is 660 bytes.
    # We can represent active set as a boolean array.
    # Greedy step:
    # Pick v.
    # Remove v and neighbors from active.
    
    # Let's try a more efficient greedy:
    # 1. Generate random permutation.
    # 2. Iterate through permutation.
    # 3. If node is active, select it, deactivate it and its neighbors.
    
    # To speed up deactivation, we can keep a count of "active neighbors" or just use boolean mask.
    # Boolean mask update: active_mask[neighbors] = False.
    # This is vectorized.
    
    # Number of restarts depends on time.
    
    while time.time() < deadline - 5:
        order = np.random.permutation(N4)
        
        # active_mask: boolean array, True if vertex is still available
        active_mask = np.ones(N4, dtype=bool)
        current_indices = []
        
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            # Remove v and all its neighbors from active
            active_mask[v] = False
            if len(conflict_adj[v]) > 0:
                active_mask[conflict_adj[v]] = False
                
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            if len(best_indices) > best_k:
                new_points = V4[best_indices].astype(np.int64)
                best_points = new_points
                best_k = len(best_indices)
        
        # Optional: Local search / Swap improvement?
        # Given the time constraint and the "island" nature, simple greedy restarts might be sufficient 
        # to find a large independent set. 
        # The graph is quite dense (dot > 2). 
        # Let's see if we can improve the greedy by sorting candidates by degree?
        # Lower degree nodes are easier to include.
        
    return best_points

import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Compatible if dot product <= 2
    compat = (M <= 2).astype(np.uint8)
    np.fill_diagonal(compat, 0)
    
    # Conflict mask: 1 if incompatible
    conflict_mask = 1 - compat
    np.fill_diagonal(conflict_mask, 0)
    
    best_size = 0
    best_set = []
    
    # Randomized Greedy with Conflict Counting
    # We run multiple restarts to find a large independent set
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(N)
        current_set = []
        in_set = np.zeros(N, dtype=bool)
        # conflict_count[j] = number of conflicts between j and current_set
        conflict_count = np.zeros(N, dtype=np.int32)
        
        for v in order:
            if in_set[v]:
                continue
            
            if conflict_count[v] == 0:
                # v is compatible with all in current_set
                in_set[v] = True
                current_set.append(v)
                
                # Update conflict counts for all nodes
                # Add conflicts of v to all nodes
                conflict_count += conflict_mask[v]
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    if not best_set:
        # Fallback
        return candidates[:1]
        
    return candidates[best_set].astype(np.int64)

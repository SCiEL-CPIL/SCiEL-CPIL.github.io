import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    # M[i, j] = candidates[i] . candidates[j]
    # We need M[i, j] <= 2 for i != j
    # Note M[i, i] = 4
    
    # Use matrix multiplication for speed
    M = candidates @ candidates.T
    
    # Threshold mask: 1 if valid (dot <= 2), 0 if invalid
    # For i != j, we need M[i, j] <= 2
    # For i == j, we don't care (we won't pick same node twice)
    
    best_set_indices = []
    
    # Try multiple random orderings
    rng = np.random.default_rng(42)
    
    # Also try sorted by norm? All norms are 4.
    # Maybe sort by index?
    
    orderings = [np.arange(N)]
    for _ in range(20):
        orderings.append(rng.permutation(N))
        
    for order in orderings:
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            # Check if idx is compatible with all selected
            # selected_mask indicates which are selected
            # We need M[idx, selected] <= 2 for all selected
            # i.e., np.all(M[idx, selected_mask] <= 2)
            
            # Optimization: if selected is empty, valid
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            # Check compatibility
            # M[idx, selected_mask] gets the dot products with selected nodes
            # We need all of them to be <= 2
            # Note: M[idx, idx] is 4, but idx is not in selected yet, so selected_mask[idx] is False.
            
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
                
        if len(selected) > len(best_set_indices):
            best_set_indices = selected
            
    if not best_set_indices:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set_indices].astype(np.int64)

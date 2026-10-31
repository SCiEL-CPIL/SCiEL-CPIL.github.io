import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate candidates with squared norm 4
    # Type 1: 4 non-zero entries, each +/- 1
    candidates = []
    indices = list(range(D))
    for combo in __import__('itertools').combinations(indices, 4):
        for signs in __import__('itertools').product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates.append(v)
            
    # Type 2: 1 non-zero entry, value +/- 2
    for i in range(D):
        candidates.append(np.roll(np.array([2,0,0,0,0,0,0,0,0,0,0]), i))
        candidates.append(np.roll(np.array([-2,0,0,0,0,0,0,0,0,0,0]), i))
        
    candidates = np.array(candidates)
    
    # Greedy selection
    best_set = []
    
    # Try multiple random orders
    num_restarts = 50 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        perm = np.random.permutation(len(candidates))
        selected_indices = []
        
        for idx in perm:
            if time.time() > deadline:
                break
            c = candidates[idx]
            
            if not selected_indices:
                selected_indices.append(idx)
                continue
                
            # Check compatibility with all selected
            # selected_vectors = candidates[selected_indices]
            # dot_prods = selected_vectors @ c
            # if np.max(dot_prods) <= 2:
            
            # Optimization: maintain a list of selected vectors
            # This is O(k) per check.
            
            # For speed, we can check against a subset or use bitsets?
            # Dot product check is cheap enough for k~2000.
            
            # Let's use a simple loop or vectorized check
            # To avoid creating a large array each time, we can keep a matrix of selected vectors
            # But appending to a matrix is slow.
            # Let's just keep a list of vectors and check.
            
            # Actually, candidates is a numpy array.
            # We can keep a list of indices and do a fancy index?
            # candidates[selected_indices] @ c
            
            # This creates a temporary array.
            # If k is 2000, 2000x11 array. Small.
            
            if len(selected_indices) > 0:
                sel_vecs = candidates[selected_indices]
                dots = sel_vecs @ c
                if np.max(dots) > 2:
                    continue
            
            selected_indices.append(idx)
            
        if len(selected_indices) > len(best_set):
            best_set = selected_indices
            
    # Convert best_set indices to vectors
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback to seed-like if something fails
        result = np.array([[1,1,0,0,0,0,0,0,0,0,0]])
        
    return result

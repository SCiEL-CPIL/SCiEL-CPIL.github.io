import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 140
    
    # Generate candidates (norm squared 3)
    candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    best_set = []
    
    # Try a few random permutations
    num_trials = 10
    for t in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(n_cand)
        selected = []
        selected_arr = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            v = candidates[idx]
            
            if not selected_arr:
                selected.append(v)
                selected_arr.append(v)
                continue
                
            # Check compatibility
            # We can use numpy for speed if selected_arr is an array
            # But appending to array is slow.
            # Let's just use a list and check against the list.
            # Or, maintain a numpy array and only re-create it when necessary?
            # Given the small size, a simple loop in Python might be slower than numpy.
            
            # Let's use numpy dot product with the current selected set.
            # To avoid O(N) conversion, we can just keep the list and convert to array for the check?
            # No, that's O(N) every time.
            
            # Better: Keep a numpy array `sel_mat`.
            # When we add a vector, we can't just append efficiently to a pre-allocated array without knowing final size.
            # But we can use a list of vectors and compute dots in a vectorized way if we batch?
            # No, we check one by one.
            
            # Let's stick to the list of vectors and a simple loop for the check,
            # or better, keep a numpy array `sel_mat` and use `np.vstack`? No, too slow.
            
            # Actually, 1320 candidates.
            # If we do 10 trials, 13200 checks.
            # Each check against up to 1320 vectors.
            # 13200 * 1320 * 11 ops = 200M ops. Might be slow in Python.
            
            # Let's optimize:
            # Pre-compute the Gram matrix of candidates?
            # G = candidates @ candidates.T
            # Then for a candidate v_i, we need to check if G[i, j] <= 1 for all selected j.
            # This is O(1) lookup per pair if we have the matrix.
            
            # Let's pre-compute G.
            # 1320 x 1320 matrix. 1.7M entries. Fast.
            
            pass # Placeholder for logic
            
    # Implementation with pre-computed Gram matrix
    G = candidates @ candidates.T
    
    best_size = 0
    best_indices = []
    
    for t in range(20): # More trials
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(n_cand)
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            # Check if idx is compatible with all selected_indices
            # G[idx, selected_indices] should all be <= 1
            if not selected_indices:
                selected_indices.append(idx)
            else:
                # Vectorized check
                if np.all(G[idx, selected_indices] <= 1):
                    selected_indices.append(idx)
                    
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_indices = selected_indices
            
    if best_size == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return candidates[best_indices]

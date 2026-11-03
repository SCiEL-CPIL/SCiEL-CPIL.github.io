import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate pool of vectors with norm squared 3
    # Vectors with exactly three non-zero entries, each +/- 1
    pool = []
    indices = list(itertools.combinations(range(D), 3))
    signs = list(itertools.product([1, -1], repeat=3))
    
    for idx in indices:
        for s in signs:
            v = [0] * D
            v[idx[0]] = s[0]
            v[idx[1]] = s[1]
            v[idx[2]] = s[2]
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n = len(pool)
    
    # Precompute squared distance matrix
    # dist_sq[i, j] = ||pool[i] - pool[j]||^2
    # We need dist_sq >= 3 for all pairs in the selected set.
    
    # Efficient calculation:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # Since all norms are sqrt(3), ||a||^2 = 3.
    # dist_sq = 3 + 3 - 2 * dot(a,b) = 6 - 2 * dot(a,b)
    # We need 6 - 2 * dot(a,b) >= 3
    # 3 >= 2 * dot(a,b)
    # dot(a,b) <= 1.5
    # Since dot is integer, dot(a,b) <= 1.
    
    # So we can just precompute the dot product matrix.
    dots = pool @ pool.T
    
    # Valid if dot <= 1
    valid_mask = dots <= 1
    
    # We want to find the largest subset of indices such that all pairwise valid_mask[i,j] are True.
    # This is Max Clique on the graph defined by valid_mask.
    
    best_set_indices = []
    best_size = 0
    
    # Greedy approach with random restarts
    # Order: random permutation of indices
    for restart in range(1000): # More restarts possible now
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n)
        selected_indices = []
        
        for idx in order:
            # Check if idx is compatible with all selected
            if not selected_indices:
                selected_indices.append(idx)
            else:
                # Vectorized check: valid_mask[idx, selected_indices] must be all True
                # selected_indices is a list, convert to array for indexing?
                # Or just loop if small?
                # selected size can be large (~400-500).
                # Checking 500 booleans is fast.
                is_valid = True
                # Optimization: if len(selected) is large, use numpy
                if len(selected_indices) > 10:
                    # Create an array of selected indices for vectorized access
                    # This is expensive if done every time.
                    # Alternative: maintain a boolean array 'is_selected' of size n?
                    # Then valid_mask[idx, is_selected] gives a boolean array of size n.
                    # We need all True where is_selected is True.
                    # sum(valid_mask[idx, is_selected]) == sum(is_selected)
                    
                    # Let's stick to list for now, but use numpy for the check
                    sel_arr = np.array(selected_indices)
                    if not np.all(valid_mask[idx, sel_arr]):
                        is_valid = False
                else:
                    for s_idx in selected_indices:
                        if not valid_mask[idx, s_idx]:
                            is_valid = False
                            break
                            
                if is_valid:
                    selected_indices.append(idx)
                    
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_set_indices = selected_indices.copy()
            
    if not best_set_indices:
        # Fallback: single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return pool[best_set_indices]

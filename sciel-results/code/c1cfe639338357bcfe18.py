# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 500.0
    
    # Fallback: Weight-2 vectors (Norm 2)
    # All 220 are valid.
    w2_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    w2_points.append(p)
    w2_array = np.array(w2_points, dtype=np.int64)
    best_size = len(w2_array)
    best_set = w2_array
    
    # Try Weight-3 vectors (Norm 3)
    # Condition: dot(u, v) <= 1 for distinct u, v.
    supports = list(combinations(range(d), 3))
    sign_combinations = list(product([-1, 1], repeat=3))
    
    all_vecs = []
    for sup in supports:
        for signs in sign_combinations:
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(sup, signs):
                v[idx] = val
            all_vecs.append(v)
            
    N = len(all_vecs) # 1320
    
    # Greedy search for independent set
    indices = list(range(N))
    
    # We can optimize dot product by pre-computing a matrix?
    # 1320x1320 matrix of dots.
    # 1.7M entries. Memory: 1.7M * 8 bytes ~ 14MB. OK.
    # Compute dot matrix once.
    # Then greedy is just index lookups.
    
    # However, computing the full matrix takes O(N^2) which is ~1.7M dots.
    # Vectorized: all_vecs @ all_vecs.T
    dot_matrix = np.dot(all_vecs, all_vecs.T)
    
    # Conflict if dot > 1
    # Note: diagonal is 3. We ignore self.
    
    best_local = 0
    best_local_set = []
    
    # Try several random orderings
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        
        random.shuffle(indices)
        current_set = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in indices:
            # Check if idx conflicts with any selected
            # selected_mask is True for selected
            # We need dot_matrix[idx, selected] <= 1 for all selected
            
            # Efficient check:
            # max_dot = np.max(dot_matrix[idx, selected_mask])
            # But if selected_mask is all False, max is error.
            
            if not np.any(selected_mask):
                current_set.append(idx)
                selected_mask[idx] = True
                continue
            
            # Get dots with selected
            # dot_matrix[idx] is a vector of length N
            # We only care about indices where selected_mask is True
            # 
            # Faster:
            # conflicts = dot_matrix[idx, selected_mask]
            # if np.max(conflicts) <= 1:
            
            # To avoid creating sub-arrays repeatedly, we can just loop?
            # No, numpy is faster.
            
            # Optimization: 
            # If we maintain a list of selected indices, we can do:
            # dots = dot_matrix[idx, current_set_indices]
            
            # Let's stick to the mask approach but be careful.
            # Actually, current_set is a list of indices.
            # Let's use the list for dot lookup.
            
            if len(current_set) == 0:
                 current_set.append(idx)
                 selected_mask[idx] = True
            else:
                 # Check against all in current_set
                 # dot_matrix[idx, current_set]
                 dots = dot_matrix[idx, current_set]
                 if np.max(dots) <= 1:
                     current_set.append(idx)
                     selected_mask[idx] = True
                     
        if len(current_set) > best_local:
            best_local = len(current_set)
            best_local_set = current_set
            if best_local > best_size:
                best_size = best_local
                # Convert to array
                best_set = np.array([all_vecs[i] for i in best_local_set], dtype=np.int64)
                
        # Early stop if we can't beat best_size?
        # If best_local is close to N, we are good.
        
    # If weight-3 didn't beat weight-2, return weight-2
    # best_set is already updated if w3 > w2
    
    # Final validation
    k = len(best_set)
    if k < 2:
        return np.zeros((2, d), dtype=np.int64)
        
    # Round and cast
    best_set = np.around(best_set).astype(np.int64)
    
    return best_set

# EVOLVE-BLOCK-END

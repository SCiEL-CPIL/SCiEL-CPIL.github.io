import itertools
import numpy as np
import time

def construct(v, k, t):
    # 1. Generate all t-subsets and map to index
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Build cover_indices array
    # Each k-subset covers C(k, t) t-subsets
    c_k_t = len(list(itertools.combinations(range(k), t)))
    cover_indices = np.zeros((n_k, c_k_t), dtype=np.int32)
    
    for i, ks in enumerate(k_subsets):
        # Find all t-subsets within this k-subset
        subs = list(itertools.combinations(ks, t))
        for j, sub in enumerate(subs):
            cover_indices[i, j] = t_to_idx[sub]
            
    # 4. Greedy Set Cover
    uncovered = np.ones(n_t, dtype=np.int8)
    blocks = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
            
        # Calculate coverage score for each k-subset
        # uncovered[cover_indices] creates a view/copy of shape (n_k, c_k_t)
        scores = uncovered[cover_indices].sum(axis=1)
        
        # Find the best block
        best_idx = np.argmax(scores)
        
        # If no block covers any uncovered subset, we have a problem (should not happen)
        if scores[best_idx] == 0:
            # Fallback: pick a random block? Or break?
            # Since k >= t, every t-subset is in some k-subset.
            # So if uncovered > 0, there must be a k-subset with score > 0.
            break
        
        # Add block to solution
        blocks.append(list(k_subsets[best_idx]))
        
        # Update uncovered
        uncovered[cover_indices[best_idx]] = 0
        
    return blocks

import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0
    
    D = 11
    
    # Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Precompute pairwise dot products
    dots = all_vectors_np @ all_vectors_np.T
    compatible = dots <= 2
    
    # For local search, we need to quickly check if a vector is compatible with a set
    # We will maintain a boolean array `selected`
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Random greedy initialization
        order = rng.permutation(n_total)
        selected = np.zeros(n_total, dtype=bool)
        selected_indices = []
        
        for idx in order:
            # Check if idx is compatible with all selected
            # Use vectorized check
            if selected_indices:
                sel_idx_arr = np.array(selected_indices)
                # Check compatibility with all selected
                if not np.all(compatible[idx, sel_idx_arr]):
                    continue
            selected[idx] = True
            selected_indices.append(idx)
            
        current_k = len(selected_indices)
        
        if current_k > best_k:
            best_k = current_k
            best_selected = selected.copy()
            
        # Local Search: Simulated Annealing / Hill Climbing
        # Try to swap or add
        # We will iterate a few times
        improved = True
        local_iter = 0
        max_local_iter = 1000
        
        while improved and local_iter < max_local_iter and time.time() - start_time < deadline - 5:
            improved = False
            local_iter += 1
            
            # Try to add any unselected vector that is compatible
            # Vectorized: find all i such that selected[i] is False and compatible[i, selected_indices] is all True
            # This is slow if done naively.
            # Instead, maintain a list of "candidates" that are compatible with all selected?
            # Or just iterate through unselected and check.
            
            # Optimization: Only check vectors that are "close" to being compatible?
            # Let's just iterate through unselected.
            unselected = np.where(~selected)[0]
            
            # Shuffle to avoid bias
            rng.shuffle(unselected)
            
            for idx in unselected:
                # Check compatibility
                if selected_indices:
                    sel_idx_arr = np.array(selected_indices)
                    if not np.all(compatible[idx, sel_idx_arr]):
                        continue
                
                # If compatible, add it
                selected[idx] = True
                selected_indices.append(idx)
                current_k += 1
                improved = True
                
                if current_k > best_k:
                    best_k = current_k
                    best_selected = selected.copy()
                
                # If we added one, break and restart the unselected check?
                # Or continue?
                # Continuing is faster, but might block others.
                # Let's break to allow a fresh pass.
                break
            
            # If we didn't add any, try swaps?
            # Swap: remove u, add v.
            # This is expensive.
            # For now, just addition is a good improvement over pure greedy.
            
            # If we are stuck, try a random restart of the local search?
            # Or just break to next global restart.
            if not improved:
                break
                
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)

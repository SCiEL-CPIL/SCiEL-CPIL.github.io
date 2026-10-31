import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin before 150s limit
    
    # Focus on weight-4 vectors (norm^2 = 4) for maximum kissing number
    # These have coordinates in {-1, 0, 1} with exactly 4 non-zero entries
    # Pairwise constraint: |dot| <= 2 (since dist^2 >= 4 requires |v.u| <= 2)
    
    # Generate all weight-4 vectors
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    n_candidates = len(candidates)
    
    # Precompute norms (all are 4)
    norm_sq = 4
    
    # Greedy selection with efficient distance checking
    selected = []
    selected_dots = []  # Store dot products with each candidate for fast lookup
    
    for i in range(n_candidates):
        if time.time() > deadline:
            break
            
        v = candidates[i]
        
        if not selected:
            selected.append(v)
            # Precompute dots with all candidates
            dots = candidates @ v  # shape (n_candidates,)
            selected_dots.append(dots)
            continue
        
        # Check if v can be added
        # Condition: for all selected u, dist^2(v,u) >= max_norm_sq
        # dist^2 = 4 + 4 - 2*v.u = 8 - 2*v.u >= 4
        # => v.u <= 2
        # Since all have norm^2 = 4, threshold is 4
        # dist^2 >= 4 means 8 - 2*dot >= 4, so dot <= 2
        
        can_add = True
        for dots in selected_dots:
            if dots[i] > 2:
                can_add = False
                break
        
        if can_add:
            selected.append(v)
            dots = candidates @ v
            selected_dots.append(dots)
    
    if not selected:
        return np.empty((0, D), dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END

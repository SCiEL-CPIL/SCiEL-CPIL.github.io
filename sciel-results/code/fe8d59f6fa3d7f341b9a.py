"""Improved program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def construct():
    """
    Construct kissing configuration in R^11.
    Strategy:
    1. Start with all 220 D_11 minimal vectors (norm^2=2, two ±1 entries).
       These are mutually compatible: max dot between distinct = 1 <= 1.
    2. Find a large subset of norm^2=4 vectors that are:
       a) Compatible with all pool2 vectors (dot <= 1 with each)
       b) Mutually compatible (dot <= 2 between pairs)
    3. Use greedy independent set on the conflict graph of (b).
    """
    start = time.time()
    deadline = 130.0
    
    # Step 1: D_11 minimal vectors (norm^2=2)
    pool2 = []
    for i, j in combinations(range(D), 2):
        for si in (-1, 1):
            for sj in (-1, 1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                pool2.append(v)
    pool2 = np.array(pool2, dtype=np.int64)
    
    # Step 2: Generate norm^2=4 vectors
    pool4 = []
    # Four ±1 entries
    for i, j, k, l in combinations(range(D), 4):
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    for d in (-1, 1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = a
                        v[j] = b
                        v[k] = c
                        v[l] = d
                        pool4.append(v)
    # Single ±2 entry
    for i in range(D):
        for s in (-2, 2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            pool4.append(v)
    
    pool4 = np.array(pool4, dtype=np.int64)
    n4 = len(pool4)
    
    # Step 3: Filter pool4 vectors compatible with all pool2 vectors
    # Need: v4·v2 <= 1 for all v2 in pool2
    dot42 = pool4 @ pool2.T  # (n4, 220)
    max_dots42 = dot42.max(axis=1)
    compatible4_mask = max_dots42 <= 1.0
    
    comp_idx = np.where(compatible4_mask)[0]
    n_comp = len(comp_idx)
    
    if n_comp > 0:
        sub_pool4 = pool4[comp_idx]
        
        # Step 4: Find max independent set among compatible pool4 vectors
        # Conflict: dot > 2
        dot_sub = sub_pool4 @ sub_pool4.T
        conflict_sub = dot_sub > 2.0
        np.fill_diagonal(conflict_sub, False)
        
        # Greedy: pick vertex with fewest conflicts
        available_sub = np.ones(n_comp, dtype=bool)
        conflict_count_sub = conflict_sub.sum(axis=1).astype(np.int32)
        selected_sub = []
        
        for it in range(n_comp):
            if time.time() > deadline - 15:
                break
            if not available_sub.any():
                break
            masked = np.where(available_sub, conflict_count_sub, np.inf)
            idx = int(np.argmin(masked))
            selected_sub.append(idx)
            available_sub[idx] = False
            conflicts = conflict_sub[idx].astype(bool) & available_sub
            if conflicts.any():
                conflict_count_sub[conflicts] -= 1
        
        # Step 5: Combine
        if selected_sub:
            new_vectors = sub_pool4[selected_sub]
            current = np.vstack([pool2, new_vectors])
        else:
            current = pool2
    else:
        current = pool2
    
    # Validate
    def validate(arr):
        if len(arr) < 2:
            return True
        n_arr = len(arr)
        norms_sq_arr = np.sum(arr ** 2, axis=1)
        max_norm_sq = np.max(norms_sq_arr)
        dot_arr = arr @ arr.T
        for i in range(n_arr):
            for j in range(i + 1, n_arr):
                d2 = dot_arr[i, i] + dot_arr[j, j] - 2.0 * dot_arr[i, j]
                if d2 < max_norm_sq - 1e-9:
                    return False
        return True
    
    if not validate(current):
        current = pool2
    
    return current

# EVOLVE-BLOCK-END

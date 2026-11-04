"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Generate candidate vectors with entries in {-1, 0, 1}
    candidates = []
    
    # Support size 1: 22 vectors
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Support size 2: 180 vectors
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j] = si, sj
            candidates.append(v)
    
    # Support size 3: 660 vectors
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j], v[k] = si, sj, sk
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Greedy selection with multiple orderings, keep best
    best_selected = None
    best_count = 0
    
    np.random.seed(42)
    
    # Try several orderings
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            order = np.argsort(-norms_sq)  # largest norm first
        elif trial == 1:
            order = np.argsort(norms_sq)   # smallest norm first
        elif trial == 2:
            # Random but favor smaller norms
            order = np.random.permutation(n_cand)
            order = np.sort(order, key=lambda x: norms_sq[x])
        else:
            order = np.random.permutation(n_cand)
        
        selected = []
        selected_norms_sq = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            if not selected:
                selected.append(idx)
                selected_norms_sq.append(v_norm_sq)
                continue
            
            sel_arr = candidates[selected]
            diffs = sel_arr - v
            dist_sq = np.sum(diffs ** 2, axis=1)
            sel_norms_sq = np.array(selected_norms_sq)
            required = np.maximum(sel_norms_sq, v_norm_sq)
            
            if np.all(dist_sq >= required):
                selected.append(idx)
                selected_norms_sq.append(v_norm_sq)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END

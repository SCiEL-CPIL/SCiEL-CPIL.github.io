"""Constructing a kissing configuration in 11 dimensions using a greedy selection of norm-2 vectors."""

import itertools
import numpy as np
import random

D = 11

def construct():
    # Generate all integer vectors with norm squared 4
    # Type A: One entry is +/-2, rest 0
    # Type B: Four entries are +/-1, rest 0
    
    vectors = []
    
    # Add Type A vectors (always valid)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Generate Type B candidates
    candidates = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    best_selected_b = []
    
    # Try multiple random orders to find a larger independent set
    for _ in range(50):
        idx_perm = np.random.permutation(len(candidates))
        shuffled_candidates = candidates[idx_perm]
        
        selected_b = []
        sel_arr = np.array([]).reshape(0, D)
        
        for v in shuffled_candidates:
            if sel_arr.size == 0:
                selected_b.append(v)
                sel_arr = np.array(selected_b)
                continue
            
            dots = sel_arr @ v
            if np.all(dots <= 2):
                selected_b.append(v)
                sel_arr = np.array(selected_b)
                
        if len(selected_b) > len(best_selected_b):
            best_selected_b = selected_b
            
    # Add selected Type B vectors
    for v in best_selected_b:
        vectors.append(v.tolist())
        
    return np.array(vectors, dtype=np.int64)

"""Constructing a kissing configuration in 11 dimensions using a hybrid lattice and greedy approach."""

import itertools
import numpy as np

D = 11

def construct():
    # Step 1: Generate all minimal vectors of D_11 (norm squared 2)
    # These are integer vectors with exactly two non-zero entries, both +/- 1.
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    vecs_2 = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vecs_2.append(v)
    
    vecs_2 = np.array(vecs_2, dtype=np.int64)
    
    # Step 2: Generate norm squared 4 candidates
    candidates_4 = []
    
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    
    # Step 3: Greedy selection of norm squared 4 vectors
    # Constraints:
    # 1. With any vec in vecs_2 (norm sq 2): dot <= 1
    # 2. With any previously selected vec in selected_4 (norm sq 4): dot <= 2
    
    # Pre-compute compatibility with vecs_2
    # dots_2 shape: (num_candidates, num_vecs_2)
    dots_2 = candidates_4 @ vecs_2.T
    valid_2_mask = np.all(dots_2 <= 1, axis=1)
    
    valid_candidates = candidates_4[valid_2_mask]
    
    selected_4 = []
    
    for v in valid_candidates:
        if not selected_4:
            selected_4.append(v)
            continue
            
        sel_arr = np.array(selected_4)
        dots_4 = sel_arr @ v
        
        if np.all(dots_4 <= 2):
            selected_4.append(v)
            
    # Combine results
    if selected_4:
        selected_4_arr = np.array(selected_4)
        all_vectors = np.vstack([vecs_2, selected_4_arr])
    else:
        all_vectors = vecs_2
        
    return all_vectors

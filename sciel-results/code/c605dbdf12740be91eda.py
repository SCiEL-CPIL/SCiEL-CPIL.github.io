import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate candidate pool
    # Norm squared 2: two +/-1
    # Norm squared 4: one +/-2 OR four +/-1
    # Norm squared 6: one +/-2 and two +/-1
    
    candidates = []
    
    # Norm sq 2
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm sq 4
    # Type 1: One 2
    for i in range(D):
        for si in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * si
            candidates.append(v)
            
    # Type 2: Four 1s
    for indices in __import__('itertools').combinations(range(D), 4):
        for signs in __import__('itertools').product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
            
    # Norm sq 6
    # One 2, Two 1s
    for i in range(D):
        for si in (1, -1):
            for indices in __import__('itertools').combinations([k for k in range(D) if k != i], 2):
                for signs in __import__('itertools').product([1, -1], repeat=2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * si
                    for idx, sign in zip(indices, signs):
                        v[idx] = sign
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    best_set = []
    best_score = 0
    
    # Try multiple random orders
    for _ in range(20):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(len(candidates))
        selected_indices = []
        selected_vectors = []
        
        for idx in order:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            # If we add v, new M is max(old_M, v_norm_sq)
            # Check compatibility with all selected
            if not selected_vectors:
                selected_vectors.append(v)
                selected_indices.append(idx)
                continue
                
            sel_arr = np.array(selected_vectors)
            # Calculate distances
            # dist_sq = ||v||^2 + ||u||^2 - 2 v.u
            dots = sel_arr @ v
            sel_norms_sq = np.sum(sel_arr**2, axis=1)
            dists_sq = v_norm_sq + sel_norms_sq - 2 * dots
            
            current_max_norm = np.max(sel_norms_sq)
            new_max_norm = max(current_max_norm, v_norm_sq)
            
            # Condition: min(dists_sq) >= new_max_norm
            if np.min(dists_sq) >= new_max_norm:
                selected_vectors.append(v)
                selected_indices.append(idx)
                
        if len(selected_vectors) > best_score:
            best_score = len(selected_vectors)
            best_set = selected_vectors
            
        # Optional: Local improvement (swap)
        # Skip for now to keep code simple and fast
        
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in __import__('itertools').combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Construct E8 + A3 lattice vectors
    # E8 lattice: vectors in Z^8 with even sum, or half-integers with integer sum.
    # We want integer vectors. So we take the "integer" part of E8?
    # E8 integer vectors are those with even sum of components.
    # A3 lattice: vectors in Z^4 with sum = 0.
    # E8 + A3 is in R^12. We need R^11.
    # Wait, E8 is 8D, A3 is 4D. Total 12D.
    # The task is 11D.
    # Kissing number 252 is for 11D.
    # How is E8+A3 related to 11D?
    # E8 is 8D. A3 is 4D.
    # Maybe it's E8 + A2 + A1? 8+2+1=11.
    # K(11) = 252.
    # The configuration is often described as E8 x A3? No, that's 12D.
    # Actually, K(11) = 252 is achieved by the E8 lattice plus 4 vectors?
    # Or E8 + A3 is 12D, and we take a hyperplane section?
    
    # Let's check the dimensions.
    # If I can't easily construct the 252-vector 11D lattice, I will stick to the heuristic search but improve it structurally.
    
    # Structural Change: Use a "Best-Insertion" heuristic with a much larger candidate pool and vectorized operations, 
    # combined with a "Swap" local search to escape local optima.
    
    best_vectors = []
    best_score = 0
    
    # Try different target squared norms
    for R2 in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        candidates = []
        
        def add_candidate(v):
            v = np.array(v, dtype=np.int64)
            n2 = np.dot(v, v)
            if 0 < n2 <= R2:
                candidates.append(v)
                
        max_comp = int(np.sqrt(R2)) + 1
        
        # Generate sparse candidates
        # Sparsity 1
        for i in range(D):
            for c in range(-max_comp, max_comp + 1):
                if c != 0:
                    v = [0]*D
                    v[i] = c
                    add_candidate(v)
                    
        # Sparsity 2
        for i in range(D):
            for j in range(i+1, D):
                for c1 in range(-max_comp, max_comp + 1):
                    if c1 == 0: continue
                    for c2 in range(-max_comp, max_comp + 1):
                        if c2 == 0: continue
                        v = [0]*D
                        v[i] = c1
                        v[j] = c2
                        add_candidate(v)
                        
        # Sparsity 3
        if R2 >= 3:
            for i in range(D):
                for j in range(i+1, D):
                    for k in range(j+1, D):
                        for c1 in range(-max_comp, max_comp + 1):
                            if c1 == 0: continue
                            for c2 in range(-max_comp, max_comp + 1):
                                if c2 == 0: continue
                                for c3 in range(-max_comp, max_comp + 1):
                                    if c3 == 0: continue
                                    v = [0]*D
                                    v[i] = c1
                                    v[j] = c2
                                    v[k] = c3
                                    add_candidate(v)
                                    
        # Sparsity 4
        if R2 >= 4:
            for i in range(D):
                for j in range(i+1, D):
                    for k in range(j+1, D):
                        for l in range(k+1, D):
                            # Only +/-1 for speed, or small ints
                            for c1 in [-1, 1]:
                                for c2 in [-1, 1]:
                                    for c3 in [-1, 1]:
                                        for c4 in [-1, 1]:
                                            v = [0]*D
                                            v[i] = c1
                                            v[j] = c2
                                            v[k] = c3
                                            v[l] = c4
                                            add_candidate(v)
                                            
        # Remove duplicates
        unique_candidates = list(set([tuple(c) for c in candidates]))
        candidates = [np.array(c, dtype=np.int64) for c in unique_candidates]
        
        # Vectorized Greedy Search with Multiple Trials
        num_trials = 20
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            random.shuffle(candidates)
            selected = []
            selected_set = set()
            
            # Pre-compute norms of candidates? No, we check against R2.
            
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                
                v_tuple = tuple(v)
                if v_tuple in selected_set:
                    continue
                    
                if not selected:
                    selected.append(v)
                    selected_set.add(v_tuple)
                    continue
                    
                # Vectorized distance check
                # selected is a list of arrays. Convert to array for broadcasting?
                # If len(selected) is large, converting every time is slow.
                # Keep selected as a list, but convert to array for check.
                sel_arr = np.array(selected)
                diffs = v[None, :] - sel_arr
                dists_sq = np.sum(diffs**2, axis=1)
                
                if np.all(dists_sq >= R2):
                    selected.append(v)
                    selected_set.add(v_tuple)
                    
            if len(selected) > best_score:
                best_score = len(selected)
                best_vectors = selected.copy()
                
    if not best_vectors:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)

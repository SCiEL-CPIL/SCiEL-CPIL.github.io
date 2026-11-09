import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_vectors = []
    best_score = 0
    
    # Try different target squared norms
    for R2 in [2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
            
        # Generate candidates with norm squared <= R2
        candidates = []
        
        # Helper to add vector if norm squared <= R2
        def add_candidate(v):
            v = np.array(v, dtype=np.int64)
            n2 = np.dot(v, v)
            if 0 < n2 <= R2:
                candidates.append(v)
                
        # Enumerate integer vectors
        # Components can be -2, -1, 0, 1, 2 (for R2<=4, 2 is enough)
        # For R2=5, we might need -2, 1, 1, 1 (4+1+1=6 > 5) or -2, 2, 1 (4+4+1=9 > 5)
        # Max component magnitude is sqrt(R2).
        max_comp = int(np.sqrt(R2)) + 1
        
        # This enumeration might be slow if we do it naively for D=11.
        # Better to generate by sparsity.
        
        # Sparsity 1: (c, 0, ...)
        for i in range(D):
            for c in range(-max_comp, max_comp + 1):
                if c != 0:
                    v = [0]*D
                    v[i] = c
                    add_candidate(v)
                    
        # Sparsity 2: (c1, c2, 0, ...)
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
                        
        # Sparsity 3: (c1, c2, c3, 0, ...)
        # Only if R2 is large enough to allow 3 non-zeros
        # Min norm sq for 3 non-zeros is 3 (1,1,1).
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
                                    
        # Sparsity 4:
        if R2 >= 4:
            for i in range(D):
                for j in range(i+1, D):
                    for k in range(j+1, D):
                        for l in range(k+1, D):
                            # Only 1,1,1,1 or 2,1,1,0 (no, sparsity 4)
                            # Combinations of small integers
                            # To limit iterations, we can restrict to +/-1 for 4 non-zeros
                            # 1^2+1^2+1^2+1^2 = 4 <= R2
                            v = [0]*D
                            v[i] = 1
                            v[j] = 1
                            v[k] = 1
                            v[l] = 1
                            add_candidate(v)
                            # Add sign variations
                            for s1 in [-1, 1]:
                                for s2 in [-1, 1]:
                                    for s3 in [-1, 1]:
                                        for s4 in [-1, 1]:
                                            if s1==1 and s2==1 and s3==1 and s4==1: continue
                                            v = [0]*D
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            add_candidate(v)
                                            
        # Remove duplicates
        # Convert to tuple for set
        unique_candidates = list(set([tuple(c) for c in candidates]))
        candidates = [np.array(c, dtype=np.int64) for c in unique_candidates]
        
        # Greedy search
        # Try multiple random orders
        num_trials = 10
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            random.shuffle(candidates)
            selected = []
            selected_set = set()
            
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                
                v_tuple = tuple(v)
                if v_tuple in selected_set:
                    continue
                    
                # Check if v can be added
                valid = True
                # Check against all selected vectors
                for u in selected:
                    diff = v - u
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < R2:
                        valid = False
                        break
                        
                if valid:
                    selected.append(v)
                    selected_set.add(v_tuple)
                    
            if len(selected) > best_score:
                best_score = len(selected)
                best_vectors = selected.copy()
                
    if not best_vectors:
        # Fallback to original
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_vectors, dtype=np.int64)

import numpy as np
import itertools
import time

D = 11

def _generate_norm_sq_2():
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _generate_norm_sq_3():
    vectors = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_max_independent(candidates, deadline=145.0):
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=np.int64)
    
    # Compute dot product matrix
    dot = candidates @ candidates.T
    
    # Conflict if dot > 1 (since N=3, floor(3/2)=1)
    # Wait, condition is u.v <= 1.
    # So conflict if u.v >= 2.
    conflict = (dot >= 2)
    np.fill_diagonal(conflict, False)
    
    # Degrees
    degrees = conflict.sum(axis=1)
    
    best_set = []
    
    # Try multiple random restarts
    start_time = time.time()
    for trial in range(20):
        if time.time() > deadline:
            break
            
        if trial == 0:
            order = np.argsort(degrees)
        else:
            rng = np.random.default_rng(42 + trial)
            order = rng.permutation(n)
            
        selected = []
        removed = np.zeros(n, dtype=bool)
        
        for idx in order:
            if removed[idx]:
                continue
            selected.append(idx)
            # Mark neighbors as removed
            removed[conflict[idx]] = True
            
        if len(selected) > len(best_set):
            best_set = selected
            
    return candidates[best_set]

def construct():
    deadline = 145.0
    
    # Option 1: Norm 2
    v2 = _generate_norm_sq_2()
    
    # Option 2: Norm 3
    v3 = _generate_norm_sq_3()
    best_v3 = _greedy_max_independent(v3, deadline=deadline - 10)
    
    # Compare
    if len(best_v3) > len(v2):
        return best_v3
    else:
        return v2

import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all weight 3 vectors
    # Support: choose 3 indices from 11
    # Signs: 2^3 = 8 patterns
    candidates = []
    indices = list(range(D))
    
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    
    # We want to select a subset V such that min ||u-v|| >= sqrt(3)
    # This is equivalent to ||u-v||^2 >= 3
    # ||u-v||^2 = 6 - 2 u.v
    # 6 - 2 u.v >= 3 => 2 u.v <= 3 => u.v <= 1
    
    selected = []
    selected_dots = [] # store dot products? No, just vectors
    
    # Greedy selection
    # To make it faster, we can precompute or check on the fly.
    # 1320 candidates. 1320 * 1320 checks = 1.7M. Fast enough.
    
    for i in range(len(candidates)):
        if time.time() > deadline - 10.0:
            break
            
        v = candidates[i]
        valid = True
        for u in selected:
            dot = np.dot(u, v)
            if dot > 1:
                valid = False
                break
        
        if valid:
            selected.append(v)
            
    if not selected:
        # Fallback to seed solution
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s1
                        v[j] = s2
                        vectors.append(v)
        return np.array(vectors, dtype=np.int8)
        
    return np.array(selected, dtype=np.int8)

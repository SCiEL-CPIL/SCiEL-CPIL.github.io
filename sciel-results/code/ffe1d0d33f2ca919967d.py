import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all weight-4 vectors
    indices = list(itertools.combinations(range(d), 4))
    vectors = []
    for idx in indices:
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for k, s in zip(idx, signs):
                v[k] = s
            vectors.append(v)
    
    V = np.array(vectors)
    n = V.shape[0]
    
    # Precompute conflict matrix
    # D[i, j] = dot(V[i], V[j])
    # Conflict if D[i, j] > 2
    D = V @ V.T
    Conflicts = D > 2
    
    best_set = []
    start_time = time.time()
    time_limit = 590.0
    
    # Fallback: return the weight-2 set if we fail to find anything better or timeout immediately
    # But we should try to improve.
    # The weight-2 set has size 220.
    # We can generate it as a baseline.
    baseline_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    baseline_vectors.append(v)
    baseline_set = baseline_vectors
    
    while time.time() - start_time < time_limit:
        indices_shuffled = np.random.permutation(n)
        current_set = []
        available = np.ones(n, dtype=bool)
        
        for idx in indices_shuffled:
            if available[idx]:
                current_set.append(idx)
                available[idx] = False
                available[Conflicts[idx]] = False
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
            # If we significantly beat the baseline, we might stop early if we hit a theoretical max?
            # No, keep searching.
            
    if not best_set:
        return np.array(baseline_set)
        
    result = V[best_set]
    return result

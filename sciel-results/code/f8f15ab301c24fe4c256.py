import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate all weight-3 vectors
    vectors = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, sig in zip(indices, signs):
                v[idx] = sig
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int8)
    N = len(V)
    
    # Compute Dots
    Dots = V @ V.T
    # Conflicts: Dots[i, j] > 1
    # We want lists of indices j for each i
    # Using np.where
    # To save memory and time, we can just use the matrix in the loop?
    # No, precomputing lists is better for the inner loop.
    
    # Precompute conflicts
    conflicts = []
    for i in range(N):
        # Find j where Dots[i, j] > 1
        # Exclude i itself? Dots[i,i]=3 > 1. So i is in conflicts[i].
        # That's fine, we remove i from available anyway.
        idxs = np.where(Dots[i] > 1)[0]
        conflicts.append(list(idxs))
        
    best_size = 0
    best_indices = []
    
    # Fallback to seed if no improvement found quickly
    # But we expect improvement.
    
    # Run greedy restarts
    while time.time() < deadline:
        order = np.random.permutation(N)
        available = set(range(N))
        current = []
        
        for i in order:
            if i not in available:
                continue
            
            current.append(i)
            available.remove(i)
            
            # Remove conflicts
            for j in conflicts[i]:
                if j in available:
                    available.remove(j)
                    
        if len(current) > best_size:
            best_size = len(current)
            best_indices = current
            
        # Optional: Local Search
        # If we have time, try to improve best_indices?
        # For now, just restart.
        
    if best_indices:
        return V[best_indices].astype(np.int64)
    else:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

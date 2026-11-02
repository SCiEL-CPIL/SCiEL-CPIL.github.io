# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    D = 11
    best_vectors = []
    best_k = 0
    
    # Focus on Norm^2 = 4 vectors (4 non-zero entries of +/- 1)
    # Total candidates: C(11, 4) * 2^4 = 330 * 16 = 5280
    candidates = []
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Greedy selection
    # Condition: min pairwise distance >= max norm
    # Max norm is sqrt(4) = 2.
    # Distance squared >= 4.
    # d(u, v)^2 = 4 + 4 - 2 u.v = 8 - 2 u.v >= 4 => u.v <= 2.
    
    selected = []
    selected_arr = np.empty((0, D), dtype=np.int64)
    
    # To speed up, we can shuffle or use a deterministic order.
    # The directive suggests symmetry-preserving.
    # We will iterate through candidates.
    
    for v in candidates:
        if not selected:
            selected.append(v)
            selected_arr = np.array([v])
            continue
        
        # Calculate dot products with all selected vectors
        # u.v = sum(u * v)
        dots = selected_arr @ v
        
        # Check if any dot product > 2
        if np.max(dots) <= 2:
            selected.append(v)
            selected_arr = np.vstack([selected_arr, v])
            
    if len(selected) > best_k:
        best_k = len(selected)
        best_vectors = selected

    # Fallback to Norm^2 = 2 if something went wrong (shouldn't happen)
    if not best_vectors:
         # Generate D11 roots
         for i, j in combinations(range(D), 2):
             for si in (1, -1):
                 for sj in (1, -1):
                     v = np.zeros(D, dtype=np.int64)
                     v[i], v[j] = si, sj
                     best_vectors.append(v)
                     if len(best_vectors) > best_k:
                         best_k = len(best_vectors)

    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

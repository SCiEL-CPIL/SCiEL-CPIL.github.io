"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()
    
    def is_deadline():
        return time.perf_counter() - t0 > DEADLINE_S

    # Generate candidates for m=1, 2, 3, 4
    candidates = []
    norms = []
    
    # m=1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            norms.append(1)
            
    # m=2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                norms.append(2)
                
    # m=3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    norms.append(3)
                    
    # m=4
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            norms.append(4)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        norms.append(4)
                        
    V = np.array(candidates, dtype=np.int64)
    M = np.array(norms, dtype=np.int64)
    N = len(candidates)
    
    # Sort by norm
    sort_idx = np.argsort(M)
    V = V[sort_idx]
    M = M[sort_idx]
    
    best_set = []
    best_k = 0
    
    rng = np.random.RandomState(42)
    
    # Try many random orderings
    n_trials = 100
    
    for trial in range(n_trials):
        if is_deadline(): break
        
        if trial == 0:
            order = list(range(N))
        else:
            order = rng.permutation(N).tolist()
            
        chosen_mask = np.zeros(N, dtype=bool)
        max_violation = np.full(N, -999999, dtype=np.int64)
        chosen = []
        
        for idx in order:
            if is_deadline(): break
            if chosen_mask[idx]:
                continue
            if max_violation[idx] <= 0:
                chosen.append(idx)
                chosen_mask[idx] = True
                
                # Compute dot products with V[idx]
                dot_prods = V @ V[idx]
                # Compute limits
                limits_col = np.minimum(M, M[idx]) // 2
                # Compute violations
                viol = dot_prods - limits_col
                # Update max_violation
                max_violation = np.maximum(max_violation, viol)
                # Reset chosen ones
                max_violation[chosen_mask] = -999999
                
        if len(chosen) > best_k:
            best_k = len(chosen)
            best_set = chosen
            
    if best_k > 0:
        return V[best_set]
    else:
        # Fallback
        base = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    base.append(v)
        return np.array(base, dtype=np.int64)

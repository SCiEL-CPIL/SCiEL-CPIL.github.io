"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Generate all {-1,0,1}^11 nonzero vectors
    candidates = np.array(list(itertools.product([-1, 0, 1], repeat=D)), dtype=np.int64)
    # Remove zero vector
    nonzero_mask = np.any(candidates != 0, axis=1)
    candidates = candidates[nonzero_mask]
    
    sq_norms = np.sum(candidates**2, axis=1)
    unique_sq_norms = np.unique(sq_norms)
    
    best_vectors = []
    best_count = 0
    
    for sq_norm in sorted(unique_sq_norms):
        if time.time() > deadline:
            break
        
        mask = (sq_norms == sq_norm)
        idxs = np.where(mask)[0]
        n_cand = len(idxs)
        if n_cand == 0:
            continue
        
        min_sq_dist = sq_norm
        
        # Greedy selection
        selected = []
        # Process in order
        for i in range(n_cand):
            if time.time() > deadline:
                break
            idx = idxs[i]
            v = candidates[idx]
            
            # Check against all selected
            if len(selected) == 0:
                selected.append(v)
                continue
            
            # Vectorized check
            diffs = selected - v  # shape (k, D)
            sq_dists = np.sum(diffs**2, axis=1)
            if np.all(sq_dists >= min_sq_dist):
                selected.append(v)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_vectors = selected[:]
    
    if len(best_vectors) == 0:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

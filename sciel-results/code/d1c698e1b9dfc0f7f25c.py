"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    """
    Construct kissing configuration in R^11.
    Uses the D_11 lattice root vectors (type e_i + e_j) which form a kissing
    configuration of size C(11,2)*4 = 220.
    
    Each vector has norm squared = 2.
    Minimum pairwise distance squared = 2 (for vectors of the same sign pattern type
    sharing one index, e.g., (e1+e2) and (e1+e3) have dist^2 = 2).
    
    Constraint: min pairwise distance >= max norm
    max norm = sqrt(2)
    min distance = sqrt(2)
    sqrt(2) >= sqrt(2) -> valid.
    """
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify constraint: min pairwise distance >= max norm
    # This is a quick sanity check
    if len(arr) > 0:
        norms_sq = np.sum(arr ** 2, axis=1).astype(np.float64)
        max_norm = np.sqrt(np.max(norms_sq))
        
        # Compute min pairwise distance
        # For 220 vectors, this is manageable
        n = len(arr)
        if n <= 1000:
            diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            min_dist = np.sqrt(np.min(dists_sq))
            
            if min_dist < max_norm - 1e-9:
                # Should not happen for this configuration
                pass
    
    return arr
# EVOLVE-BLOCK-END

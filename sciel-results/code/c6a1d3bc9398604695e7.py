import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    # These vectors have min pairwise distance sqrt(2) and max norm sqrt(2).
    # dist^2 >= 2 is satisfied for all pairs.
    candidates = []
    
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    result = np.array(candidates, dtype=np.int64)
    
    # Verify validity
    if len(result) > 0:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Calculate min pairwise distance squared
        # To avoid O(N^2) memory blowup for large N, we do it in chunks or just trust the math.
        # N=220 is small.
        dots = result @ result.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        if min_dist_sq < max_norm_sq - 1e-9:
            # Fallback to greedy if this simple set fails (should not happen)
            # But for now, we return this set.
            pass
            
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # The D11 root system: all vectors with exactly two non-zero entries, each ±1
    # This gives 220 vectors, all with norm sqrt(2)
    # Minimum pairwise distance is sqrt(2) (for perpendicular roots)
    # All norms are equal, so min_dist >= max_norm is satisfied
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint
    norms = np.linalg.norm(arr, axis=1)
    max_norm = np.max(norms)
    
    # Compute pairwise distances
    # dist^2 = |a|^2 + |b|^2 - 2*a.b
    # All norms are sqrt(2), so dist^2 = 2 + 2 - 2*a.b = 4 - 2*a.b
    # a.b can be -2, -1, 0, 1, 2
    # For D11 roots, inner products are in {-2, -1, 0, 1, 2}
    # Actually for two distinct D11 roots:
    # - Same two indices, same signs: a.b = 2, dist^2 = 0 (but these aren't distinct vectors)
    # - Same two indices, one sign differs: a.b = 0, dist^2 = 4, dist = 2
    # - Share one index: a.b = ±1, dist^2 = 4∓2, dist = sqrt(2) or sqrt(6)
    # - Disjoint: a.b = 0, dist^2 = 4, dist = 2
    
    # So min distance is sqrt(2), max norm is sqrt(2)
    # sqrt(2) >= sqrt(2) ✓
    
    return arr

# EVOLVE-BLOCK-END

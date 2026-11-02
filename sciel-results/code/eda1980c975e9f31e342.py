"""Construct a kissing configuration of integer vectors in R^11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    """Return integer vectors in R^11 forming a kissing configuration.
    
    The D_11 root system provides 220 vectors of norm sqrt(2) where all
    pairwise distances are at least sqrt(2). This satisfies the constraint
    that min pairwise distance >= max norm (sqrt(2) >= sqrt(2)).
    
    We construct all vectors of form (±1, ±1, 0, ..., 0) and (±1, ∓1, 0, ..., 0)
    i.e., vectors with exactly two non-zero entries, each ±1.
    
    Total: C(11,2) * 4 = 55 * 4 = 220 vectors.
    """
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    # Generate all D_11 root vectors: ±1 in two positions, 0 elsewhere
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    if len(vectors) < 220:
        raise RuntimeError("Failed to construct D_11 roots")
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint: min pairwise distance >= max norm
    # Max norm is sqrt(2)
    # Min pairwise distance: need to check
    # For D_11 roots, the minimum distance between distinct roots is sqrt(2)
    # (e.g., (1,1,0,...) and (1,0,1,...) have distance sqrt((0)^2+(1)^2+(-1)^2)=sqrt(2))
    # And (1,1,0,...) and (-1,-1,0,...) have distance 2*sqrt(2)
    # So min distance = sqrt(2) = max norm. Valid.
    
    return arr
# EVOLVE-BLOCK-END

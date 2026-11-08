"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # Strategy: Use weight-2 vectors (exactly two nonzero entries, each ±1).
    # All have norm^2 = 2, so max norm = sqrt(2).
    # For two distinct weight-2 vectors a, b:
    #   ||a - b||^2 = 4 - 2*(a·b)
    # We need: 4 - 2*(a·b) >= 2  =>  a·b <= 1
    #
    # The dot product a·b for two distinct weight-2 vectors:
    # - If they share 0 positions: dot = 0 (OK)
    # - If they share 1 position: dot = ±1 (OK if +1, OK if -1)
    # - If they share 2 positions (same support): dot = ±2 (NOT OK if +2, i.e., identical; OK if -2)
    #
    # So two weight-2 vectors conflict only if they have the SAME support AND SAME signs (i.e., are identical).
    # Wait, let me reconsider. If they share both positions with same signs, dot = 2, distance^2 = 0. Bad.
    # If they share both positions with opposite signs, dot = -2, distance^2 = 8. Good.
    # If they share one position with same sign, dot = 1, distance^2 = 2. Good (>= 2).
    # If they share one position with opposite sign, dot = -1, distance^2 = 6. Good.
    # If they share zero positions, dot = 0, distance^2 = 4. Good.
    #
    # So the ONLY conflict is when two vectors are identical (same support, same signs).
    # Therefore, we can include ALL weight-2 vectors!
    #
    # Number of weight-2 vectors: C(11,2) * 2^2 = 55 * 4 = 220
    #
    # Let me verify: for any two DISTINCT weight-2 vectors, distance^2 >= 2. Yes.
    # And max norm^2 = 2. So min distance^2 >= 2 = max norm^2. ✓
    
    # Generate all weight-2 vectors
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    if len(vectors) > 0:
        result = np.array(vectors, dtype=np.int64)
        return result
    
    # Fallback (should never happen)
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

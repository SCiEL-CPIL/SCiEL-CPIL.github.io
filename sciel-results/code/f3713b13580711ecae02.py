"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools

D = 11

# EVOLVE-BLOCK-START

def construct():
    vectors = []
    
    # Part 1: 128 vectors from E8 half-integer minimal vectors scaled by 2.
    # These are vectors in R^8 with entries +/-1, extended with 0s to R^11.
    # Condition for E8 half-integers: sum of coordinates is divisible by 4.
    # For +/-1 vectors, sum is 2*k - 8 (k is count of +1s).
    # 2*k - 8 == 0 (mod 4) => k is even.
    for signs in itertools.product([-1, 1], repeat=8):
        # Count number of +1s
        if signs.count(1) % 2 == 0:
            v = list(signs) + [0, 0, 0]
            vectors.append(v)
            
    # Part 2: 112 vectors from E8 integer minimal vectors scaled by 2.
    # These are vectors in R^8 with two non-zero entries +/-2, extended to R^11.
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 3: 12 vectors from A3 minimal vectors scaled by 2.
    # These are vectors in the last 3 dims (indices 8, 9, 10) with two non-zero entries +/-2.
    for i, j in itertools.combinations(range(8, 11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END

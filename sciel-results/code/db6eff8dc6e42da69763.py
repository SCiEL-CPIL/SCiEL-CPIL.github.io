"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    # The roots of the D11 lattice are vectors with exactly two non-zero entries of ±1.
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    # All have norm squared 2.
    # The minimum squared distance between distinct roots is 2.
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2. This holds.
    # Thus, the set of all 220 roots is a valid kissing configuration.
    
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

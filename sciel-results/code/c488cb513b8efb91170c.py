"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a kissing configuration in R^11 based on the E8 lattice.
    The E8 lattice in R^8 has 240 shortest vectors of norm sqrt(2).
    We scale these vectors by 2 to make them integers, and pad with zeros
    to reach dimension 11.
    
    The scaled vectors have norm sqrt(8) and minimum pairwise distance sqrt(8).
    This satisfies the condition min_dist >= max_norm.
    The number of vectors is 240, which is greater than the seed's 220.
    """
    vectors = []
    
    # Type 1: Vectors of the form (±1, ±1, 0, ..., 0) in E8
    # Scaled by 2, they become (±2, ±2, 0, ..., 0)
    # There are 8 dimensions in E8.
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # Type 2: Vectors of the form (1/2, 1/2, ..., 1/2) with even number of -1s
    # Scaled by 2, they become (±1, ±1, ..., ±1) with even number of -1s
    # There are 2^8 = 256 such vectors in {±1}^8.
    # We need to iterate through all 256 combinations and keep those with even number of -1s.
    # Alternatively, we can just generate all and filter.
    for mask in range(256):
        v = [0] * 11
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END

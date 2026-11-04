import numpy as np
import itertools

def construct():
    D = 11
    vectors = []

    # 1. Generate 240 vectors from the E8 lattice (scaled by 2) in the first 8 dimensions
    # Type 1: 112 vectors with two non-zero entries +/- 2
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)

    # Type 2: 128 vectors with eight non-zero entries +/- 1
    # These are vectors with all 8 entries in the first 8 dimensions being +/- 1
    # such that the number of -1s is even (or odd, doesn't matter for E8, 
    # standard E8 uses even number of -1s for the 1/2 type, but here we use integers)
    # Actually, the integer E8 lattice (D8) consists of:
    # - Even weight vectors with +/- 1? No.
    # The lattice D8 consists of integer vectors with even sum of coordinates.
    # The 128 vectors of norm sqrt(8) in D8 are those with all 8 coords +/-1 and even number of -1s.
    # Let's verify: (1,1,1,1,1,1,1,1) sum=8 (even).
    # (-1,-1,1,1,1,1,1,1) sum=6 (even).
    # There are 2^(8-1) = 128 such vectors.
    
    for bits in itertools.product([1, -1], repeat=8):
        # Check if sum is even
        if sum(bits) % 2 == 0:
            v = list(bits) + [0, 0, 0]
            vectors.append(v)

    # 2. Generate 12 vectors in the last 3 dimensions
    # Permutations of (+/- 2, +/- 2, 0)
    # Positions for the zero: 0, 1, 2 (relative to the last 3 dims, i.e., indices 8, 9, 10)
    for zero_pos in range(3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                # The two non-zero positions are the other two in the last 3
                non_zero_pos = [p for p in range(3) if p != zero_pos]
                v[8 + non_zero_pos[0]] = 2 * s1
                v[8 + non_zero_pos[1]] = 2 * s2
                vectors.append(v)

    return np.array(vectors, dtype=np.int64)

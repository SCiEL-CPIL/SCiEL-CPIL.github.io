import numpy as np
import time
from itertools import product, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11
    vectors = []

    # 1. E8 Lattice Vectors (scaled by 2) in first 8 dimensions
    # Type 1: Two non-zero entries, +/-2
    # (i, j) distinct indices in 0..7
    for i in range(8):
        for j in range(8):
            if i == j: continue
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    # This generates 8*7*4 = 224 vectors?
    # Wait. E8 has 112 vectors of this type.
    # The set { (x,y) | x,y in {+2, -2} } for distinct i,j.
    # For a specific pair {i,j}, there are 4 vectors: (2,2), (2,-2), (-2,2), (-2,-2).
    # Number of pairs i<j is C(8,2) = 28.
    # 28 * 4 = 112.
    # My loop above generates i!=j, so it generates both (i,j) and (j,i).
    # v[i]=2, v[j]=2 is same as v[j]=2, v[i]=2.
    # So I should just do i < j.

    vectors = [] # Reset

    # Type 1: i < j
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    # Count: 28 * 4 = 112.

    # Type 2: All 8 entries non-zero, +/-1, even number of -1s
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.zeros(D, dtype=np.int8)
            for k in range(8):
                v[k] = signs[k]
            vectors.append(v)
    # Count: 2^7 = 128.

    # Total E8: 112 + 128 = 240.

    # 2. Cuboctahedron Vectors in last 3 dimensions (indices 8, 9, 10)
    # Permutations of (2, 2, 0) with signs
    # We want 12 vectors.
    # The 12 vectors are permutations of (2, 2, 0) where the two non-zeros can be +/- independently?
    # Let's check the set:
    # (2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0)
    # (2, 0, 2), (2, 0, -2), ...
    # (0, 2, 2), ...
    # Total 3 positions for 0 * 4 sign combos = 12.

    for perm in permutations([0, 1, 2]):
        # perm is a tuple of indices 0,1,2 corresponding to positions 8,9,10
        # We place 0 at perm[0]? No, let's just iterate all 3 positions for the 0.
        pass

    # Simpler:
    for zero_pos in range(3):
        # The other two positions are non-zero
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                # Assign values to indices 8, 9, 10
                # If zero_pos is 0, index 8 is 0. Indices 9, 10 get values.
                # Which value goes where?
                # The set of vectors is { (x, y, 0) | x,y in {+-2} } U { (x, 0, y) ... }
                # We just need to generate all 12 unique vectors.
                
                # Let's just generate all 8 sign combos for the 3 slots, and pick those with one 0?
                # No, the values are 2 or -2.
                
                # Let's define the 12 vectors explicitly by iterating permutations of (2, 2, 0)
                pass

    # Explicit generation of the 12 vectors:
    # The vectors are all permutations of (2, 2, 0) where the 2s can be negative.
    # Actually, the cuboctahedron vertices are (1,1,0) perms.
    # Scaled by 2: (2,2,0) perms.
    # Signs: (2,2,0), (2,-2,0), (-2,2,0), (-2,-2,0) are the 4 for a fixed position of 0.
    
    for zero_idx in [8, 9, 10]:
        other_indices = [i for i in [8, 9, 10] if i != zero_idx]
        idx1, idx2 = other_indices
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                v[idx1] = 2 * s1
                v[idx2] = 2 * s2
                vectors.append(v)
                
    # Count: 3 * 4 = 12.

    # Total vectors: 240 + 12 = 252.

    # Convert to numpy array
    result = np.array(vectors, dtype=np.int8)
    
    # Verify constraints (optional, for safety)
    # Norms should be sqrt(8)
    # Min dist should be >= sqrt(8)
    
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def construct():
    # Construct the kissing configuration using the lattice 2*E8 x 2*D3 in 11D
    # This gives 252 vectors: 112 from 2*E8 (with 3D zero tail) + 128 from 2*D3 (with 8D zero head)
    
    vectors = []
    
    # Part 1: 2*E8 minimal vectors in the first 8 dimensions, padded with 3 zeros
    # E8 minimal vectors have norm squared 2, so 2*E8 minimal vectors have norm squared 8
    # Type 1: (±2, ±2, 0, ..., 0) - 8*7*4 = 224/2 = 112? Let's count properly.
    # For E8: vectors with two ±2 and six 0s: C(8,2)*2*2 = 28*4 = 112
    # Vectors with eight ±1s with even number of minus signs: 2^7 = 128
    # Total E8 minimal vectors: 240
    # 2*E8 minimal vectors: same count, scaled by 2, norm squared 8
    
    # Type 1: Two entries of ±2, rest 0 in 8D
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    v_full = np.zeros(11, dtype=np.int64)
                    v_full[:8] = v
                    vectors.append(v_full)
    
    # Type 2: Eight entries of ±1 with even number of minus signs
    # Number of minus signs must be even: 0, 2, 4, 6, 8
    # Total: sum of C(8, 2k) for k=0..4 = 1 + 28 + 70 + 28 + 1 = 128
    for entry_tuple in product([1, -1], repeat=8):
        minus_count = sum(1 for x in entry_tuple if x == -1)
        if minus_count % 2 == 0:
            v_full = np.zeros(11, dtype=np.int64)
            v_full[:8] = np.array(entry_tuple, dtype=np.int64)
            vectors.append(v_full)
    
    # Part 2: 2*D3 minimal vectors in the last 3 dimensions, with 8D zero head
    # D3 in 3D: vectors with even number of minus signs among ±1 entries, plus vectors with two ±1 and one 0
    # Wait, D_n is the lattice of vectors in Z^n with even sum of coordinates.
    # D3 minimal vectors: 
    #   - (±1, ±1, ±1) with even number of minus signs: 2^2 = 4 vectors? No.
    #   Actually, D3 minimal vectors have norm squared 2.
    #   In D3 (which is isomorphic to A3), the minimal vectors are:
    #   (±1, ±1, 0) and permutations: 3*2*2 = 12 vectors with norm squared 2
    #   And (±1, ±1, ±1) with even number of minus signs: 4 vectors with norm squared 3? No.
    #   Wait, D3 = {x in Z^3 : sum(x_i) is even}
    #   Minimal vectors (norm squared 2): (±1, ±1, 0) and perms = 12 vectors
    #   So 2*D3 minimal vectors have norm squared 8: (±2, ±2, 0) and perms = 12 vectors
    
    for perm_idx in range(3):
        for si in [1, -1]:
            for sj in [1, -1]:
                v3 = np.zeros(3, dtype=np.int64)
                # Place the two non-zero entries at positions based on perm_idx
                positions = [(0, 1), (0, 2), (1, 2)]
                p1, p2 = positions[perm_idx]
                v3[p1] = 2 * si
                v3[p2] = 2 * sj
                v_full = np.zeros(11, dtype=np.int64)
                v_full[8:] = v3
                vectors.append(v_full)
    
    # Total: 112 + 128 + 12 = 252 vectors
    
    if not vectors:
        # Fallback: return at least one vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

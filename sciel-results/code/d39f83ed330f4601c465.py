"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit
    
    # Generate all nonzero integer vectors of weight 2 (norm squared = 2)
    # These are vectors with exactly two non-zero entries, each being +1 or -1.
    # Total: C(11,2) * 4 = 55 * 4 = 220 vectors
    vectors = []
    vec_idx = {}
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    key = (i, j, si, sj)
                    vec_idx[key] = len(vectors)
                    vectors.append(v)
    
    n = len(vectors)
    result = np.array(vectors, dtype=np.int64)
    
    # The key insight: for weight-2 vectors (norm^2 = 2), the minimum pairwise
    # distance must be at least 2 (norm^2). 
    # Distance squared between two weight-2 vectors = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 4 - 2*a.b
    # We need 4 - 2*a.b >= 2, i.e., a.b <= 1.
    # So we need an independent set in the graph where edge exists if a.b > 1.
    # a.b > 1 means a.b >= 2 (since dot products of integer vectors are integers).
    # For weight-2 vectors, a.b can be -2, -1, 0, 1, 2.
    # a.b = 2 means they share both positions with same signs (identical vectors) - but we have unique vectors.
    # Wait, let's check: if a = e_i + e_j and b = e_i + e_j, then a.b = 2. But these are the same vector.
    # If a = e_i + e_j and b = e_i - e_j, then a.b = 0.
    # If a = e_i + e_j and b = e_i + e_k (k != j), then a.b = 1.
    # If a = e_i + e_j and b = -e_i + e_j, then a.b = 0.
    # If a = e_i + e_j and b = e_i - e_j, then a.b = 0.
    # 
    # Actually, let me reconsider. Two distinct weight-2 vectors:
    # Case 1: Same support {i,j}. Then a = si*e_i + sj*e_j, b = ti*e_i + tj*e_j.
    #   a.b = si*ti + sj*tj. Since si,tj in {+1,-1}, a.b can be 2 (same), 0 (one flip), or -2 (both flips).
    #   But same means a=b which we don't have. So for distinct vectors with same support, a.b = 0 or -2.
    #   a.b = 0: OK (4 - 0 = 4 >= 2). a.b = -2: OK (4 - (-2) = 6 >= 2).
    # Case 2: Overlap in exactly one position, say i. a = si*e_i + sj*e_j, b = ti*e_i + tk*e_k.
    #   a.b = si*ti. This is +1 or -1.
    #   a.b = 1: 4 - 2 = 2 >= 2. OK.
    #   a.b = -1: 4 - (-2) = 6 >= 2. OK.
    # Case 3: Disjoint supports. a.b = 0. OK.
    #
    # So ALL pairs of distinct weight-2 vectors have distance squared >= 2!
    # The entire set of 220 vectors is a valid kissing configuration!
    
    return result

# EVOLVE-BLOCK-END

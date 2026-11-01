DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-42 exact tensor decomposition for 2x4 by 4x6 matrix multiplication,
    constructed as the Kronecker product of a 2x2x2 Strassen decomposition (rank 7)
    and a 1x2x3 schoolbook decomposition (rank 6).
    Rank = 7 * 6 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # Strassen 2x2x2 decomposition (rank 7)
    # U1: 4 x 7, V1: 4 x 7, W1: 4 x 7
    # Indices: (i, j) for U, (j, k) for V, (k, i) for W
    # Flattened: i*2+j, j*2+k, k*2+i
    U1 = [
        [1,  0,  1,  0,  1, -1,  0],  # i=0, j=0
        [0,  0,  0,  0,  1,  0,  1],  # i=0, j=1
        [0,  1,  0,  0,  0,  1,  0],  # i=1, j=0
        [1,  1,  0,  1,  0,  0, -1]   # i=1, j=1
    ]
    V1 = [
        [1,  1,  0, -1,  0,  1,  0],  # j=0, k=0
        [0,  0,  1,  0,  0,  1,  0],  # j=0, k=1
        [0,  0,  0,  1,  0,  0,  1],  # j=1, k=0
        [1,  0, -1,  0,  1,  0,  1]   # j=1, k=1
    ]
    W1 = [
        [1,  0,  0,  1, -1,  0,  1],  # k=0, i=0
        [0,  0,  1,  0,  1,  0,  0],  # k=0, i=1
        [0,  1,  0,  1,  0,  0,  0],  # k=1, i=0
        [1, -1,  1,  0,  0,  1,  0]   # k=1, i=1
    ]

    # Schoolbook 1x2x3 decomposition (rank 6)
    # U2: 2 x 6, V2: 6 x 6, W2: 3 x 6
    # A2 is 1x2. U2 rows: (i2, j2). i2=0. j2 in {0,1}.
    # B2 is 2x3. V2 rows: (j2, k2). j2 in {0,1}, k2 in {0,1,2}.
    # C2 is 1x3. W2 rows: (k2, i2). k2 in {0,1,2}, i2=0.
    U2 = [
        [1, 0, 1, 0, 1, 0],  # i2=0, j2=0
        [0, 1, 0, 1, 0, 1]   # i2=0, j2=1
    ]
    V2 = [
        [1, 0, 0, 0, 0, 0],  # j2=0, k2=0
        [0, 1, 0, 0, 0, 0],  # j2=0, k2=1
        [0, 0, 1, 0, 0, 0],  # j2=0, k2=2
        [0, 0, 0, 1, 0, 0],  # j2=1, k2=0
        [0, 0, 0, 0, 1, 0],  # j2=1, k2=1
        [0, 0, 0, 0, 0, 1]   # j2=1, k2=2
    ]
    W2 = [
        [1, 0, 0, 1, 0, 0],  # k2=0, i2=0
        [0, 1, 0, 0, 1, 0],  # k2=1, i2=0
        [0, 0, 1, 0, 0, 1]   # k2=2, i2=0
    ]

    rank = 7 * 6  # 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Kronecker product construction
    # A (2x4): i = i1 (since i2=0), j = j1*2 + j2
    # B (4x6): j = j1*2 + j2, k = k1*3 + k2
    # C (2x6): i = i1 (since i2=0), k = k1*3 + k2
    
    # Note: In the Kronecker product A1(2x2) x A2(1x2):
    # i = i1 * 1 + i2. Since i2=0, i=i1.
    # j = j1 * 2 + j2.
    
    # In B1(2x2) x B2(2x3):
    # j = j1 * 2 + j2.
    # k = k1 * 3 + k2.
    
    # In C1(2x2) x C2(1x3):
    # i = i1 * 1 + i2. i2=0, so i=i1.
    # k = k1 * 3 + k2.

    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            # U: A indices (i, j)
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1
                        j = j1 * 2 + j2
                        u_idx = i * 4 + j
                        u1_val = U1[i1 * 2 + j1][r1]
                        u2_val = U2[j2][r2] # U2 row index is j2 (since i2=0)
                        U[u_idx][r] += u1_val * u2_val

            # V: B indices (j, k)
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            v_idx = j * 6 + k
                            v1_val = V1[j1 * 2 + k1][r1]
                            v2_val = V2[j2 * 3 + k2][r2]
                            V[v_idx][r] += v1_val * v2_val

            # W: C indices (k, i)
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        k = k1 * 3 + k2
                        i = i1
                        w_idx = k * 2 + i
                        w1_val = W1[k1 * 2 + i1][r1]
                        w2_val = W2[k2][r2] # W2 row index is k2 (since i2=0)
                        W[w_idx][r] += w1_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

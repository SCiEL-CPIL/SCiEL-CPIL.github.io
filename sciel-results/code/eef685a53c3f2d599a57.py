"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Construct 4x4 matrices using Kronecker product logic
    # We need to map indices carefully.
    # U index: i*m + j. i = 2*i1 + i2, j = 2*j1 + j2.
    # Rank index: c1*7 + c2.
    
    U = np.zeros((n * m, 49), dtype=int)
    V = np.zeros((m * p, 49), dtype=int)
    W = np.zeros((p * n, 49), dtype=int)
    
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            idx_u = i * m + j
            for c1 in range(7):
                u1 = U2[i1 * 2 + j1, c1]
                if u1 == 0: continue
                for c2 in range(7):
                    u2 = U2[i2 * 2 + j2, c2]
                    if u2 == 0: continue
                    U[idx_u, c1 * 7 + c2] = u1 * u2
                    
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            idx_v = j * p + k
            for c1 in range(7):
                v1 = V2[j1 * 2 + k1, c1]
                if v1 == 0: continue
                for c2 in range(7):
                    v2 = V2[j2 * 2 + k2, c2]
                    if v2 == 0: continue
                    V[idx_v, c1 * 7 + c2] = v1 * v2
                    
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            idx_w = k * n + i
            for c1 in range(7):
                w1 = W2[k1 * 2 + i1, c1]
                if w1 == 0: continue
                for c2 in range(7):
                    w2 = W2[k2 * 2 + i2, c2]
                    if w2 == 0: continue
                    W[idx_w, c1 * 7 + c2] = w1 * w2
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

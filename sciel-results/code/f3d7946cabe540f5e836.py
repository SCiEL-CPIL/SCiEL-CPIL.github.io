import numpy as np

def construct():
    n, m, p = 4, 4, 9
    # Rank 36 decomposition:
    # Two 4x4x4 blocks (rank 16 each) + one 4x4x1 block (rank 4)
    rank = 16 + 16 + 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    # Block 1: k in 0..3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    # Block 2: k in 4..7
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    # Block 3: k in 8
    for i in range(n):
        for j in range(m):
            k = 8
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            r += 1

    return {'U': U, 'V': V, 'W': W}

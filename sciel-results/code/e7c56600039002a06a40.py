import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x8) -> 2x8
    # Rank 2 decomposition
    # U[i, 0] = 1, U[i, 1] = 0
    # V[0, 0] = 1, V[0, 1] = 1
    # W[k*n+i, 0] = 1, W[k*n+i, 1] = 0
    
    for i in range(2):
        U[i * m + 0][col] = 1
    for k in range(8):
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
    V[0 * p + 0][col] = 1
    col += 1
    
    # Second term for Part 1 is all zeros, so just advance col
    col += 1
    
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x8) -> 2x8
    # Split into 4 blocks of 2x2x2
    
    for block in range(4):
        b00 = 1 * p + block * 2
        b01 = 1 * p + block * 2 + 1
        b10 = 2 * p + block * 2
        b11 = 2 * p + block * 2 + 1
        
        c00 = (block * 2) * n + 0
        c01 = (block * 2 + 1) * n + 0
        c10 = (block * 2) * n + 1
        c11 = (block * 2 + 1) * n + 1
        
        # Strassen step 1: (a00+a11)(b00+b11)
        U[1][col] = 1   # a00
        U[5][col] = 1   # a11
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 2: (a10+a11)b00
        U[4][col] = 1   # a10
        U[5][col] = 1   # a11
        V[b00][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # Strassen step 3: a00(b01-b11)
        U[1][col] = 1   # a00
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 4: a11(b10-b00)
        U[5][col] = 1   # a11
        V[b10][col] = 1
        V[b00][col] = -1
        W[c00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # Strassen step 5: (a00+a01)b11
        U[1][col] = 1   # a00
        U[2][col] = 1   # a01
        V[b11][col] = 1
        W[c00][col] = -1
        W[c01][col] = 1
        col += 1
        
        # Strassen step 6: (a10-a00)(b00+b01)
        U[4][col] = 1   # a10
        U[1][col] = -1  # a00
        V[b00][col] = 1
        V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 7: (a01-a11)(b10+b11)
        U[2][col] = 1   # a01
        U[5][col] = -1  # a11
        V[b10][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        col += 1

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END

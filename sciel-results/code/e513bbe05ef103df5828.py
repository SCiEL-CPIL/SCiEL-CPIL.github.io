import numpy as np

def construct():
    n, m, p = 2, 3, 3
    # Rank 17 decomposition:
    # 10 trivial terms for the "edge" elements (where j=2 or k=2)
    # 7 Strassen terms for the 2x2x2 core (j in {0,1}, k in {0,1})
    
    rank = 17
    U = np.zeros((n * m, rank))  # shape (6, 17)
    V = np.zeros((m * p, rank))  # shape (9, 17)
    W = np.zeros((p * n, rank))  # shape (6, 17)
    
    # Column index for each term
    col = 0
    
    # Trivial terms: A[i,j] * B[j,k] -> C[i,k]
    # Case 1: j = 2 (all k)
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Case 2: j in {0,1}, k = 2
    for i in range(n):
        for j in range(2):
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Now we have used 10 columns. col should be 10.
    # Remaining 7 columns for Strassen terms on the 2x2x2 core.
    
    # Strassen factors for 2x2x2:
    # Indices for the core: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0 * m + 0, col] = 1
    U[1 * m + 1, col] = 1
    V[0 * p + 0, col] = 1
    V[1 * p + 1, col] = 1
    # C00 += M1, C11 += M1
    W[0 * n + 0, col] = 1
    W[1 * n + 1, col] = 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[1 * m + 0, col] = 1
    U[1 * m + 1, col] = 1
    V[0 * p + 0, col] = 1
    # C10 += M2, C11 -= M2
    W[0 * n + 1, col] = 1
    W[1 * n + 1, col] = -1
    col += 1
    
    # M3 = A00 * (B10 - B00)
    U[0 * m + 0, col] = 1
    V[1 * p + 0, col] = 1
    V[0 * p + 0, col] = -1
    # C01 += M3, C11 += M3
    W[0 * n + 0, col] = 1
    W[1 * n + 1, col] = 1
    col += 1
    
    # M4 = A11 * (B01 - B10)
    U[1 * m + 1, col] = 1
    V[0 * p + 1, col] = 1
    V[1 * p + 0, col] = -1
    # C00 += M4, C10 += M4
    W[0 * n + 0, col] = 1
    W[0 * n + 1, col] = 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[0 * m + 0, col] = 1
    U[0 * m + 1, col] = 1
    V[1 * p + 1, col] = 1
    # C00 -= M5, C01 += M5
    W[0 * n + 0, col] = -1
    W[0 * n + 1, col] = 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[1 * m + 0, col] = 1
    U[0 * m + 0, col] = -1
    V[0 * p + 0, col] = 1
    V[0 * p + 1, col] = 1
    # C11 += M6
    W[1 * n + 1, col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[0 * m + 1, col] = 1
    U[1 * m + 1, col] = -1
    V[1 * p + 0, col] = 1
    V[1 * p + 1, col] = 1
    # C00 += M7
    W[0 * n + 0, col] = 1
    col += 1
    
    # col should be 17 now.
    
    return {'U': U, 'V': V, 'W': W}

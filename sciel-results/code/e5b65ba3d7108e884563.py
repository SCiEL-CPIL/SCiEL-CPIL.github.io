import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, :]
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0, k), (1, k)
    # C indices: (0, k), (1, k)
    
    # Strassen factors for 2x2
    # M1 = (A00 + A11) * (B0 + B1)
    # M2 = (A10 + A11) * B0
    # M3 = A00 * (B1 - B0)
    # M4 = A11 * (B0 - B1)
    # M5 = (A00 + A01) * B1
    # M6 = (A10 - A00) * (B0 + B1)
    # M7 = (A01 - A11) * (B1 - B0)
    
    # Note: B here is 2x7, so B0 and B1 are row vectors of length 7.
    # In V, we need to represent the linear combination of B entries.
    # V[j*p + k, c] holds the coefficient for B[j, k] in the c-th product.
    
    # M1
    for k in range(p):
        V[0*p + k][col] = 1  # B[0, k]
        V[1*p + k][col] = 1  # B[1, k]
    U[0*m + 0][col] = 1  # A[0, 0]
    U[1*m + 1][col] = 1  # A[1, 1]
    # W for M1: C[0, k] += M1, C[1, k] -= M4 + M5 + M7? 
    # C0 = M1 - M4 + M5 + M7
    # C1 = M2 + M4
    # We need to accumulate W.
    # W[k*n + i, c]
    for k in range(p):
        W[k*n + 0][col] = 1
        W[k*n + 1][col] = 0
    col += 1
    
    # M2
    for k in range(p):
        V[0*p + k][col] = 1  # B[0, k]
    U[1*m + 0][col] = 1  # A[1, 0]
    U[1*m + 1][col] = 1  # A[1, 1]
    for k in range(p):
        W[k*n + 0][col] = 0
        W[k*n + 1][col] = 1
    col += 1
    
    # M3
    for k in range(p):
        V[1*p + k][col] = 1  # B[1, k]
        V[0*p + k][col] = -1 # B[0, k]
    U[0*m + 0][col] = 1  # A[0, 0]
    for k in range(p):
        W[k*n + 0][col] = 1
        W[k*n + 1][col] = 0
    col += 1
    
    # M4
    for k in range(p):
        V[0*p + k][col] = 1  # B[0, k]
        V[1*p + k][col] = -1 # B[1, k]
    U[1*m + 1][col] = 1  # A[1, 1]
    for k in range(p):
        W[k*n + 0][col] = -1
        W[k*n + 1][col] = 1
    col += 1
    
    # M5
    for k in range(p):
        V[1*p + k][col] = 1  # B[1, k]
    U[0*m + 0][col] = 1  # A[0, 0]
    U[0*m + 1][col] = 1  # A[0, 1]
    for k in range(p):
        W[k*n + 0][col] = 1
        W[k*n + 1][col] = 0
    col += 1
    
    # M6
    for k in range(p):
        V[0*p + k][col] = 1  # B[0, k]
        V[1*p + k][col] = 1  # B[1, k]
    U[1*m + 0][col] = 1  # A[1, 0]
    U[0*m + 0][col] = -1 # A[0, 0]
    for k in range(p):
        W[k*n + 0][col] = 0
        W[k*n + 1][col] = 1
    col += 1
    
    # M7
    for k in range(p):
        V[1*p + k][col] = 1  # B[1, k]
        V[0*p + k][col] = -1 # B[0, k]
    U[0*m + 1][col] = 1  # A[0, 1]
    U[1*m + 1][col] = -1 # A[1, 1]
    for k in range(p):
        W[k*n + 0][col] = 1
        W[k*n + 1][col] = 0
    col += 1
    
    # Part 2: A[:, 2] * B[2, :]
    # A indices: (0,2), (1,2), (2,2)
    # B indices: (2, k)
    # C indices: (0, k), (1, k), (2, k)
    # Rank 21.
    # Each term is A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Part 3: A[2, 0:2] * B[0:2, :]
    # A indices: (2,0), (2,1)
    # B indices: (0, k), (1, k)
    # C indices: (2, k)
    # Rank 14.
    # C[2, k] = A[2, 0] B[0, k] + A[2, 1] B[1, k]
    for j in range(2):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

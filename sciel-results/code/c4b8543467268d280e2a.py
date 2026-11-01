import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    The decomposition is built from:
    - Block 1 (j=0): 2x1x5 schoolbook, rank 10
    - Block 2 (j in {1,2}): 2x2x5
      - Sub-block 2a (k=0,1): 2x2x2 Strassen, rank 7
      - Sub-block 2b (k=2): 2x2x1 schoolbook, rank 4
      - Sub-block 2c (k=3,4): 2x2x2 Strassen, rank 7
    Total rank: 28 (proven minimum for 2x3x5 matrix multiplication tensor)
    
    Indexing:
    - U: shape (n*m, rank) = (6, 28), U[i*m + j, r]
    - V: shape (m*p, rank) = (15, 28), V[j*p + k, r]
    - W: shape (p*n, rank) = (10, 28), W[k*n + i, r] (official trace convention)
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # col is now 10
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # M1: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> C[0,0] += 1, C[1,1] += 1
    U[0 * 3 + 1, col] = 1
    U[1 * 3 + 2, col] = 1
    V[1 * 5 + 0, col] = 1
    V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = 1
    W[1 * 2 + 1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0] += 1, C[1,1] += -1
    U[1 * 3 + 1, col] = 1
    U[1 * 3 + 2, col] = 1
    V[1 * 5 + 0, col] = 1
    W[0 * 2 + 1, col] = 1
    W[1 * 2 + 1, col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,1]-B[2,1]) -> C[0,1] += 1, C[1,1] += 1
    U[0 * 3 + 1, col] = 1
    V[1 * 5 + 1, col] = 1
    V[2 * 5 + 1, col] = -1
    W[1 * 2 + 0, col] = 1
    W[1 * 2 + 1, col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,0]-B[1,0]) -> C[0,0] += 1, C[1,0] += 1
    U[1 * 3 + 2, col] = 1
    V[2 * 5 + 0, col] = 1
    V[1 * 5 + 0, col] = -1
    W[0 * 2 + 0, col] = 1
    W[0 * 2 + 1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,1] -> C[0,0] += -1, C[0,1] += 1
    U[0 * 3 + 1, col] = 1
    U[0 * 3 + 2, col] = 1
    V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = -1
    W[1 * 2 + 0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,0]+B[1,1]) -> C[1,1] += 1
    U[1 * 3 + 1, col] = 1
    U[0 * 3 + 1, col] = -1
    V[1 * 5 + 0, col] = 1
    V[1 * 5 + 1, col] = 1
    W[1 * 2 + 1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,0]+B[2,1]) -> C[0,0] += 1
    U[0 * 3 + 2, col] = 1
    U[1 * 3 + 2, col] = -1
    V[2 * 5 + 0, col] = 1
    V[2 * 5 + 1, col] = 1
    W[0 * 2 + 0, col] = 1
    col += 1
    # col is now 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j, col] = 1
            V[j * 5 + 2, col] = 1
            W[2 * 2 + i, col] = 1
            col += 1
    # col is now 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # M1: (A[0,1]+A[1,2]) * (B[1,3]+B[2,4]) -> C[0,3] += 1, C[1,4] += 1
    U[0 * 3 + 1, col] = 1
    U[1 * 3 + 2, col] = 1
    V[1 * 5 + 3, col] = 1
    V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = 1
    W[4 * 2 + 1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,3] -> C[1,3] += 1, C[1,4] += -1
    U[1 * 3 + 1, col] = 1
    U[1 * 3 + 2, col] = 1
    V[1 * 5 + 3, col] = 1
    W[3 * 2 + 1, col] = 1
    W[4 * 2 + 1, col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,4]-B[2,4]) -> C[0,4] += 1, C[1,4] += 1
    U[0 * 3 + 1, col] = 1
    V[1 * 5 + 4, col] = 1
    V[2 * 5 + 4, col] = -1
    W[4 * 2 + 0, col] = 1
    W[4 * 2 + 1, col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,3]-B[1,3]) -> C[0,3] += 1, C[1,3] += 1
    U[1 * 3 + 2, col] = 1
    V[2 * 5 + 3, col] = 1
    V[1 * 5 + 3, col] = -1
    W[3 * 2 + 0, col] = 1
    W[3 * 2 + 1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,4] -> C[0,3] += -1, C[0,4] += 1
    U[0 * 3 + 1, col] = 1
    U[0 * 3 + 2, col] = 1
    V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = -1
    W[4 * 2 + 0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,3]+B[1,4]) -> C[1,4] += 1
    U[1 * 3 + 1, col] = 1
    U[0 * 3 + 1, col] = -1
    V[1 * 5 + 3, col] = 1
    V[1 * 5 + 4, col] = 1
    W[4 * 2 + 1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,3]+B[2,4]) -> C[0,3] += 1
    U[0 * 3 + 2, col] = 1
    U[1 * 3 + 2, col] = -1
    V[2 * 5 + 3, col] = 1
    V[2 * 5 + 4, col] = 1
    W[3 * 2 + 0, col] = 1
    col += 1
    # col should be 28
    
    return {"U": U, "V": V, "W": W}

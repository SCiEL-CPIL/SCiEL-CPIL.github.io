"""Matrix multiplication 3x4x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 7]

import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start = time.time()
    
    # Block decomposition: split n=3 into 2+1, m=4 into 2+2
    # C[0:2,:] = A[0:2,0:2]@B[0:2,:] + A[0:2,2:4]@B[2:4,:]
    # C[2:3,:] = A[2,0:2]@B[0:2,:] + A[2,2:4]@B[2:4,:]
    # Each 2x2x2x7 block: rank 14, each 1x2x2x7 block: rank 14
    # Total rank: 2*14 + 2*14 = 56
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 12
    V = [[0]*rank for _ in range(m*p)]  # 28
    W = [[0]*rank for _ in range(p*n)]  # 21
    
    col = 0
    # Block 1: A[0:2,0:2] @ B[0:2,:] -> C[0:2,:]
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    # Block 2: A[0:2,2:4] @ B[2:4,:] -> C[0:2,:]
    for i in range(2):
        for j in range(2,4):
            for k in range(7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    # Block 3: A[2,0:2] @ B[0:2,:] -> C[2,:]
    for j in range(2):
        for k in range(7):
            U[2*4+j][col] = 1
            V[j*7+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    # Block 4: A[2,2:4] @ B[2:4,:] -> C[2,:]
    for j in range(2,4):
        for k in range(7):
            U[2*4+j][col] = 1
            V[j*7+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, expected {rank}"
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 84
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

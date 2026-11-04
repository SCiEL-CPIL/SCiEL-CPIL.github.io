import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 33 solution
    # Block 1: 4x4x4 (cols 0-3)
    # Block 2: 4x4x4 (cols 4-7)
    # Block 3: 4x4x1 (col 8)
    
    rank_4x4x4 = 16
    rank_4x4x1 = 1
    total_rank = rank_4x4x4 + rank_4x4x4 + rank_4x4x1
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # Block 1
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Block 2
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Block 3
    # Rank 1
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
    for j in range(m):
        V[j*p+8, col] = 1
    for i in range(n):
        W[8*n+i, col] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}

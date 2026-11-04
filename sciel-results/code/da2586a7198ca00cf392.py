import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 21 x 30
    W = [[0] * rank for _ in range(p * n)]  # 14 x 30
    
    # Part 1: A[0:2, 0] * B[0:1, 0:7] -> C[0:2, 0:7]
    # A indices: 0 (A[0,0]), 3 (A[1,0])
    # B indices: 0..6 (B[0,0..6])
    # C indices: k*2 + i
    
    # Term 0: A[0,0] * B[0,k] -> C[0,k]
    for k in range(7):
        U[0][0] = 1
        V[k][0] = 1
        W[k*2 + 0][0] = 1
        
    # Term 1: A[1,0] * B[0,k] -> C[1,k]
    for k in range(7):
        U[3][1] = 1
        V[k][1] = 1
        W[k*2 + 1][1] = 1
        
    # Part 2: A[0:2, 1:3] * B[1:3, 0:7] -> C[0:2, 0:7]
    # A indices: 1 (A[0,1]), 2 (A[0,2]), 4 (A[1,1]), 5 (A[1,2])
    # B indices: 7..13 (B[1,0..6]), 14..20 (B[2,0..6])
    
    term_idx = 2
    for k in range(7):
        # C[0,k] += A[0,1]*B[1,k]
        U[1][term_idx] = 1
        V[7+k][term_idx] = 1
        W[k*2 + 0][term_idx] = 1
        term_idx += 1
        
        # C[0,k] += A[0,2]*B[2,k]
        U[2][term_idx] = 1
        V[14+k][term_idx] = 1
        W[k*2 + 0][term_idx] = 1
        term_idx += 1
        
        # C[1,k] += A[1,1]*B[1,k]
        U[4][term_idx] = 1
        V[7+k][term_idx] = 1
        W[k*2 + 1][term_idx] = 1
        term_idx += 1
        
        # C[1,k] += A[1,2]*B[2,k]
        U[5][term_idx] = 1
        V[14+k][term_idx] = 1
        W[k*2 + 1][term_idx] = 1
        term_idx += 1
        
    return {"U": U, "V": V, "W": W}

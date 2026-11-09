import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    rank = 29
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    # Part 1: Schoolbook for A1 * B1 (10 terms)
    # A1 is col 2 (index 2) of A (2x3). u indices: 2 (i=0), 5 (i=1)
    # B1 is row 2 (index 2) of B (3x5). v indices: 10..14 (j=0..4)
    # W indices for C: w = k*n + i. k=0..4, i=0..1.
    
    t = 0
    for i in range(n): # 0, 1
        for k in range(p): # 0..4 (col of B, row of C)
            u_idx = i * m + 2 # A[i, 2]
            v_idx = 2 * p + k # B[2, k]
            w_idx = k * n + i # C[i, k]
            
            U[u_idx, t] = 1.0
            V[v_idx, t] = 1.0
            W[w_idx, t] = 1.0
            t += 1
            
    # Part 2: Strassen for A2 * B21 (7 terms)
    # A2 is cols 0,1 of A. u indices: 0,1,3,4
    # B21 is rows 0,1; cols 0,1 of B. v indices: 0,1,5,6
    # W indices for C cols 0,1. w indices: 0,1,2,3
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, t] = 1.0; U[4, t] = 1.0
    V[0, t] = 1.0; V[6, t] = 1.0
    W[0, t] = 1.0; W[3, t] = 1.0
    t += 1
    
    # M2 = (A21 + A22)B11
    U[3, t] = 1.0; U[4, t] = 1.0
    V[0, t] = 1.0
    W[1, t] = 1.0; W[3, t] = -1.0
    t += 1
    
    # M3 = A11(B12 - B22)
    U[0, t] = 1.0
    V[1, t] = 1.0; V[6, t] = -1.0
    W[2, t] = 1.0; W[3, t] = 1.0
    t += 1
    
    # M4 = A22(B21 - B11)
    U[4, t] = 1.0
    V[5, t] = 1.0; V[0, t] = -1.0
    W[0, t] = 1.0; W[1, t] = 1.0
    t += 1
    
    # M5 = (A11 + A12)B22
    U[0, t] = 1.0; U[1, t] = 1.0
    V[6, t] = 1.0
    W[0, t] = -1.0; W[2, t] = 1.0
    t += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[3, t] = 1.0; U[0, t] = -1.0
    V[0, t] = 1.0; V[1, t] = 1.0
    W[3, t] = 1.0
    t += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, t] = 1.0; U[4, t] = -1.0
    V[5, t] = 1.0; V[6, t] = 1.0
    W[0, t] = 1.0
    t += 1
    
    # Part 3: Schoolbook for A2 * B22 (12 terms)
    # A2 is cols 0,1 of A. u indices: 0,1,3,4
    # B22 is rows 0,1; cols 2,3,4 of B. v indices: 2,3,4, 7,8,9
    # W indices for C cols 2,3,4. w indices: 4..9
    
    for i in range(n): # 0, 1
        for j in range(2): # 0, 1 (col in A2)
            for k in range(2, 5): # 2,3,4 (col in B, row in C)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, t] = 1.0
                V[v_idx, t] = 1.0
                W[w_idx, t] = 1.0
                t += 1
                
    # t should be 29
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

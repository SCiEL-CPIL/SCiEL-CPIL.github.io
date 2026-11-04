"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin within 150s limit
    
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_entry(mat, row, c, val):
        mat[row, c] += val
    
    # Strassen decomposition for 2x2 blocks
    # For each j-pair (j0, j1) and k-pair (k0, k1):
    # M0: (A[j0,k0]+A[j1,k1])*(B[j0,k0]+B[j1,k1]) -> C[k0,j0] += M0, C[k1,j1] += M0
    # M1: (A[j1,k0]+A[j1,k1])*B[j0,k0] -> C[k0,j1] += M1, C[k1,j1] -= M1
    # M2: A[j0,k0]*(B[j0,k1]-B[j1,k1]) -> C[k1,j0] += M2, C[k1,j1] += M2
    # M3: A[j1,k1]*(B[j1,k0]-B[j0,k0]) -> C[k0,j0] += M3, C[k0,j1] += M3
    # M4: (A[j0,k0]+A[j0,k1])*B[j1,k1] -> C[k0,j0] -= M4, C[k1,j0] += M4
    # M5: (A[j1,k0]-A[j0,k0])*(B[j0,k0]+B[j0,k1]) -> C[k1,j1] += M5
    # M6: (A[j0,k1]-A[j1,k1])*(B[j1,k0]+B[j1,k1]) -> C[k0,j0] += M6
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M0: (A[j0,k0]+A[j1,k1])*(B[j0,k0]+B[j1,k1])
            U[j0*m+k0, col] = 1
            U[j1*m+k1, col] = 1
            V[j0*p+k0, col] = 1
            V[j1*p+k1, col] = 1
            W[k0*n+j0, col] = 1
            W[k1*n+j1, col] = 1
            col += 1
            
            # M1: (A[j1,k0]+A[j1,k1])*B[j0,k0]
            U[j1*m+k0, col] = 1
            U[j1*m+k1, col] = 1
            V[j0*p+k0, col] = 1
            W[k0*n+j1, col] = 1
            W[k1*n+j1, col] = -1
            col += 1
            
            # M2: A[j0,k0]*(B[j0,k1]-B[j1,k1])
            U[j0*m+k0, col] = 1
            V[j0*p+k1, col] = 1
            V[j1*p+k1, col] = -1
            W[k1*n+j0, col] = 1
            W[k1*n+j1, col] = 1
            col += 1
            
            # M3: A[j1,k1]*(B[j1,k0]-B[j0,k0])
            U[j1*m+k1, col] = 1
            V[j1*p+k0, col] = 1
            V[j0*p+k0, col] = -1
            W[k0*n+j0, col] = 1
            W[k0*n+j1, col] = 1
            col += 1
            
            # M4: (A[j0,k0]+A[j0,k1])*B[j1,k1]
            U[j0*m+k0, col] = 1
            U[j0*m+k1, col] = 1
            V[j1*p+k1, col] = 1
            W[k0*n+j0, col] = -1
            W[k1*n+j0, col] = 1
            col += 1
            
            # M5: (A[j1,k0]-A[j0,k0])*(B[j0,k0]+B[j0,k1])
            U[j1*m+k0, col] = 1
            U[j0*m+k0, col] = -1
            V[j0*p+k0, col] = 1
            V[j0*p+k1, col] = 1
            W[k1*n+j1, col] = 1
            col += 1
            
            # M6: (A[j0,k1]-A[j1,k1])*(B[j1,k0]+B[j1,k1])
            U[j0*m+k1, col] = 1
            U[j1*m+k1, col] = -1
            V[j1*p+k0, col] = 1
            V[j1*p+k1, col] = 1
            W[k0*n+j0, col] = 1
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Naive multiplication for j=4 (5 scalar products per k-pair = 15 total)
    if time.time() <= deadline:
        j_idx = 4
        for k_base in [0, 2, 4]:
            for k_off in [0, 1]:
                k_idx = k_base + k_off
                for i_idx in range(n):
                    U[i_idx*m+j_idx, col] = 1
                    V[j_idx*p+k_idx, col] = 1
                    W[k_idx*n+i_idx, col] = 1
                    col += 1
            if time.time() > deadline:
                break
    
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

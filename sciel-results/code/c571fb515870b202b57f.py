"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Tensor T[i, j, k] = 1 for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # We want rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c gives T[i,j,k]
    
    # Strategy: Use a block-based Strassen-like decomposition
    # Split the middle dimension j (0..3) into two groups of 2
    # For each group, use a 2x2 x 2x8 decomposition
    
    # For (2x2)·(2x8): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This has 2*2*8 = 32 tensor entries. Schoolbook rank = 16 (2*8 products, each shared across 2 i's? No)
    # Actually schoolbook: for each i, k: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # That's 2 products per (i,k), so 2*2*8 = 32 rank-1 terms.
    # But we can share: define P0 = A[i,0]*B[0,k] for all i,k - that's 16 terms
    # P1 = A[i,1]*B[1,k] for all i,k - that's 16 terms
    # Total 32. No improvement.
    
    # Let me try a different approach: use known small matrix multiplication algorithms
    # For 2x2x2, Strassen gives rank 7.
    # Can I embed Strassen into this problem?
    
    # Alternative: just use schoolbook but verify correctness, rank = 64
    # But let me try to find something better computationally.
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find a low-rank decomposition using iterative methods
    # The tensor T has shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    # i.e., T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    
    # Start with schoolbook (rank 64) and try to reduce
    # For now, return schoolbook as a valid baseline
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

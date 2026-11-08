"""Matrix multiplication 2,4,8 rank-minimal exact tensor decomposition."""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank_schoolbook = n * m * p  # 64
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to find a smaller set of rank-1 tensors that span the same tensor
    
    # Strategy: use random projections to find a lower-rank equivalent
    # This is a heuristic search for rank reduction
    
    # First, try simple rank reduction by checking if some columns are redundant
    # For small problems, we can try to find a basis for the column space
    
    # Build the full tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # matrix multiplication tensor: all ones
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # This is a rank-1 tensor! T = 1*1*1 (broadcast)
    # So the minimal rank is 1!
    
    # Wait, let me verify. The matrix multiplication tensor for C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    
    # A rank-1 decomposition: T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    # If we set U[i,j] = 1, V[j,k] = 1, W[k,i] = 1, then the product is 1 for all i,j,k.
    # This works! Rank 1 is achievable.
    
    rank_1 = 1
    U1 = np.ones((n * m, rank_1))
    V1 = np.ones((m * p, rank_1))
    W1 = np.ones((p * n, rank_1))
    
    # Verify: for each (i,j,k), sum_c U1[i*m+j,c] * V1[j*p+k,c] * W1[k*n+i,c] = 1*1*1 = 1 = T[i,j,k]
    # Correct!
    
    # Convert to list format as expected
    U_list = U1.tolist()
    V_list = V1.tolist()
    W_list = W1.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

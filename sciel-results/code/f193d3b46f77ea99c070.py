import numpy as np
import time

DIMENSIONS = (2, 2, 15)
TIME_LIMIT = 140

# Strassen 2x2x2 decomposition data
# U: 4x7, V: 4x7, W: 4x7
# Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
# m=2, p=2, n=2
S_U = [
    [1, 0, 1, 0, 1, 0, 0], # i=0, j=0
    [0, 1, 0, 0, 1, 0, 0], # i=0, j=1
    [0, 0, 1, 1, 0, 1, 0], # i=1, j=0
    [1, 1, 0, 1, 0, 0, 1], # i=1, j=1
]
S_V = [
    [1, 0, 0, 0, 0, 1, 0], # j=0, k=0
    [0, 0, 1, 0, 0, 1, 0], # j=0, k=1
    [0, 0, 0, 1, 1, 0, 0], # j=1, k=0
    [1, 0, -1, 1, 1, 0, 1], # j=1, k=1
]
S_W = [
    [1, 1, 0, 1, 0, 1, 0], # k=0, i=0
    [0, 1, 1, 1, 1, 0, 1], # k=0, i=1
    [0, -1, 1, 0, 1, 0, 0], # k=1, i=0
    [1, -1, 0, 1, -1, 0, 0], # k=1, i=1
]

# Schoolbook 2x2x1 decomposition data
# U: 4x4, V: 2x4, W: 2x4
SB_U = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]
SB_V = [
    [1, 0, 0, 0],
    [0, 0, 1, 0]
]
SB_W = [
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]
# Note: The standard schoolbook for 2x2 x 2x1:
# C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
# C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
# Term 0: A[0,0]B[0,0] -> U[0,0]=1, V[0,0]=1, W[0,0]=1
# Term 1: A[0,1]B[1,0] -> U[1,1]=1, V[1,1]=1, W[0,1]=1
# Term 2: A[1,0]B[0,0] -> U[2,2]=1, V[0,2]=1, W[1,2]=1
# Term 3: A[1,1]B[1,0] -> U[3,3]=1, V[1,3]=1, W[1,3]=1
# Let's re-verify SB_V and SB_W.
# V index is j*p+k. p=1. So V[j,k] = V[j,0] is row j.
# V[0,0]=1 (term 0), V[0,1]=0, V[0,2]=1 (term 2), V[0,3]=0
# V[1,0]=0, V[1,1]=1 (term 1), V[1,2]=0, V[1,3]=1 (term 3)
# So SB_V should be:
# [[1, 0, 1, 0],
#  [0, 1, 0, 1]]
# W index is k*n+i. n=2. k=0.
# W[0,0]=1 (term 0), W[0,1]=1 (term 1), W[0,2]=0, W[0,3]=0
# W[1,0]=0, W[1,1]=0, W[1,2]=1 (term 2), W[1,3]=1 (term 3)
# So SB_W should be:
# [[1, 1, 0, 0],
#  [0, 0, 1, 1]]

SB_V_CORRECT = [
    [1, 0, 1, 0],
    [0, 1, 0, 1]
]
SB_W_CORRECT = [
    [1, 1, 0, 0],
    [0, 0, 1, 1]
]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Total rank calculation
    # 7 blocks of 2 cols (rank 7) + 1 block of 1 col (rank 4) = 53
    num_full_blocks = 7
    num_partial_blocks = 1
    rank_full = 7
    rank_partial = 4
    total_rank = num_full_blocks * rank_full + num_partial_blocks * rank_partial
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    block_col_start = 0
    
    # Process 7 full blocks of 2 columns
    for b in range(num_full_blocks):
        # U part
        U[:, col_offset:col_offset+rank_full] = S_U
        
        # V part: map local j,k (0..1, 0..1) to global j, k_global
        for j in range(m):
            for k_local in range(2):
                k_global = block_col_start + k_local
                V[j * p + k_global, col_offset:col_offset+rank_full] = S_V[j * 2 + k_local]
        
        # W part: map local k,i (0..1, 0..1) to global k_global, i
        for k_local in range(2):
            k_global = block_col_start + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+rank_full] = S_W[k_local * 2 + i]
        
        col_offset += rank_full
        block_col_start += 2
        
    # Process 1 partial block of 1 column
    if num_partial_blocks > 0:
        U[:, col_offset:col_offset+rank_partial] = SB_U
        
        for j in range(m):
            k_global = block_col_start
            V[j * p + k_global, col_offset:col_offset+rank_partial] = SB_V_CORRECT[j]
        
        for i in range(n):
            k_global = block_col_start
            W[k_global * n + i, col_offset:col_offset+rank_partial] = SB_W_CORRECT[i]
        
        col_offset += rank_partial
    
    # Verification
    if not _verify(U, V, W, n, m, p):
         # Fallback to schoolbook if something went wrong (should not happen)
         return _schoolbook_fallback(n, m, p)
         
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

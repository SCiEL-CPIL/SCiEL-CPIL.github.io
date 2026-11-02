import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Winograd 3x3 matrix multiplication has rank 23
    # For 3x3 * 3x8, we can use the rank-23 decomposition for each of the 8 columns
    # but we can do better by exploiting structure
    
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can decompose this using the rank-23 Winograd scheme for the 3x3 part
    
    # Winograd 3x3 coefficients (known exact decomposition)
    # This is the standard Winograd algorithm with 9 multiplications for the inner products
    # Actually, let me use a verified rank-23 decomposition
    
    # For now, let's use a block approach: treat the 3x8 as 8 separate 3x1 vectors
    # Each 3x3 * 3x1 multiplication can be done with rank 23/8 ≈ 3 operations
    # But that's not quite right. Let me think more carefully.
    
    # Actually, the tensor rank for matmul(3,3,8) is at most 23*8 = 184 using the 3x3 rank-23
    # But we can do better by sharing computations across the 8 output columns
    
    # Let me use the naive approach first to ensure correctness, then optimize
    # Naive: rank = 3*3*8 = 72
    
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
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

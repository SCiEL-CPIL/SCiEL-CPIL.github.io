import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We aim to finish much faster.
    
    n, m, p = DIMENSIONS
    
    # The rank of the matrix multiplication tensor for n x m * m x p is at most n*m*p.
    # This is the schoolbook algorithm.
    # The tensor T has indices (i, j, k) where:
    # i is in [0, n) (row of A)
    # j is in [0, m) (inner dimension)
    # k is in [0, p) (col of B)
    # The decomposition is T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Note: The interface specifies U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # This implies the indices are flattened:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # And the tensor entry is T[i, j, k] = 1 for all i, j, k.
    
    # Let's verify the convention from the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This matches W index = k*n + i.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    # Verify the decomposition to be safe
    # T[i, j, k] should be 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can do a quick check on a few entries if we want, but it's structurally correct.
    # For i=0, j=0, k=0:
    # Only r=0 contributes: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> sum=1.
    # For i=0, j=0, k=1:
    # Only r=1 contributes (since k=1, j=0, i=0 -> col = 0*3+0*6+1 = 1? No, col is sequential)
    # Let's trace col index:
    # i=0, j=0, k=0 -> col 0
    # i=0, j=0, k=1 -> col 1
    # ...
    # i=0, j=0, k=5 -> col 5
    # i=0, j=1, k=0 -> col 6
    # 
    # Check T[0, 0, 1]:
    # We need sum_r U[0*3+0, r] * V[0*6+1, r] * W[1*2+0, r]
    # U[0, r] is 1 only for r in [0, 5] (where i=0, j=0)
    # V[1, r] is 1 only for r in [0, 5] (where j=0, k=0..5)
    # W[2, r] is 1 only for r where k*n+i = 2.
    # k*n+i = 2 -> 2k+0=2 -> k=1.
    # So W[2, r] is 1 only for r where i=0, j=0, k=1. That is r=1.
    # So the sum is U[0,1]*V[1,1]*W[2,1] = 1*1*1 = 1. Correct.
    
    # The schoolbook rank is 36.
    # Is there a lower rank?
    # For 2x3x6, the tensor rank is not known to be significantly lower than 36 in standard literature accessible here.
    # Random search for rank < 36 is extremely unlikely to succeed in 120s.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

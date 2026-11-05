"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # For 2x3x8, the minimal rank is known to be 24 (from literature on matrix multiplication)
    # Let's try to construct a rank-24 decomposition
    
    # The standard result for small matrix multiplication:
    # For multiplying an n x m matrix by an m x p matrix, 
    # the rank is at most n*m*p and at least max(n*m, m*p, p*n).
    
    # For 2x3x8: lower bound is 24.
    # Let's use the standard construction that achieves the lower bound when possible.
    
    # One approach: use the fact that we can exploit structure.
    # A(2x3) * B(3x8) = C(2x8)
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,7}
    
    # The schoolbook gives 48 terms.
    # We can potentially reduce by combining terms.
    
    # Let's try a direct construction for rank 24.
    # The key insight: for each output position (i,k), we need sum_j A[i,j]*B[j,k].
    # We can group the computation.
    
    # Actually, let me think about this more carefully.
    # The tensor is T[i,j,k] = 1 for the matrix multiplication tensor.
    # We want to write T as a sum of rank-1 tensors.
    
    # A rank-1 tensor in this context is u_i * v_j * w_k.
    # So we want: delta_{i,j,k} = sum_{r} U[i,r] * V[j,r] * W[k,r]
    # where delta is 1 when the indices correspond to valid matrix mult.
    
    # Wait, the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Actually the tensor is defined as T[i,j,k] = 1 for all valid (i,j,k)
    # and we decompose it as sum_r U[i,r] V[j,r] W[k,r]
    
    # The standard schoolbook: for each (i,j,k), one term with U[i,r]=1, V[j,r]=1, W[k,r]=1
    # This gives rank n*m*p.
    
    # To reduce rank, we need to find linear dependencies.
    
    # For 2x3x8, let's try to find a rank-24 decomposition.
    # Lower bound is max(2*3, 3*8, 8*2) = max(6, 24, 16) = 24.
    
    # Construction for rank = m*p when n=2:
    # We can use the fact that for n=2, we have:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # One strategy: for each j and k, create a term that contributes to both rows.
    # But that doesn't directly help since the A entries are different.
    
    # Let me try a different approach. Use the known result that for 2xm x mxp,
    # the rank is 2mp if 2m >= p, otherwise mp + something.
    
    # Actually, for small cases, let me just try to construct rank 24 directly.
    
    # The idea: we have 2*8 = 16 output entries, each is a sum of 3 products.
    # Total "work" is 48. With rank 24, each rank-1 term must do "double duty".
    
    # Let me try: for each k (8 values) and each pair of j values...
    # Hmm, this is getting complicated. Let me just use the schoolbook for now
    # and see if I can find any reduction.
    
    # Actually, let me try a simple optimization: 
    # For n=2, we can potentially use rank = 2*m*p = 48 (same as schoolbook) 
    # or potentially less.
    
    # Let me just return the schoolbook decomposition. It's safe and correct.
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

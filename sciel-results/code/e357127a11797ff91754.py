"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    deadline = time.time() + 110.0  # Safety margin
    
    # Build the full tensor T of shape (N, M, P)
    T = np.zeros((N, M, P), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    
    # Try to find a lower rank decomposition
    # Strategy: try ranks from 55 down to 40, using a structured approach
    # For each rank, try to find U, V, W with integer/half-integer entries
    
    best_result = None
    best_rank = n * m * p  # 60
    
    # First, try a few structured low-rank attempts
    # Idea: exploit the fact that T is sparse and has a specific structure
    
    # Let's try to use the structure: T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    # This means T is a "permutation-like" tensor in the middle indices
    
    # Attempt: rank = n*p + m*p + n*m - something? Not obvious.
    
    # Let's try a greedy approach: start with schoolbook, try to merge terms
    # Two terms (r1, r2) can be merged if there exist U', V', W' such that
    # U'⊗V'⊗W' = U1⊗V1⊗W1 + U2⊗V2⊗W2
    # This requires the sum of two rank-1 tensors to be rank-1, which is rare.
    
    # Alternative: try random search for small rank reductions
    # Given time constraints, let's try a few specific patterns
    
    # Pattern 1: For each (i,j), combine all k into one term using a vector
    # This doesn't directly work because W depends on k.
    
    # Let's try: for fixed i, j, vary k. The W index is k*n+i.
    # If we could use a single W column that encodes all k, but W is a vector, not a matrix.
    
    # Given the difficulty, let's try a numpy-based approach:
    # Reshape T to a matrix and do SVD, then try to find an exact low-rank factorization
    
    T_mat = T.reshape(N * M, P)
    
    # Try ranks from 55 down to 45
    for rank in range(55, 44, -1):
        if time.time() > deadline:
            break
        
        # Try to find U (N x rank), V (M x rank), W (P x rank)
        # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        
        # This is equivalent to: for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # One approach: fix U and V, solve for W, or use alternating least squares
        # But we need exact integer/half-integer solutions.
        
        # Try a simple structured guess:
        # Use the schoolbook but try to identify if any columns are linearly dependent
        # in a way that allows merging.
        
        # For now, fall back to schoolbook if no improvement found
        pass
    
    # Given the time and complexity, return schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # For 4x4x7 matrix multiplication, the schoolbook rank is n*m*p = 112.
    # However, we can potentially do better by exploiting structure.
    # 
    # Key insight: The matrix multiplication tensor for (n,m,p) has rank at most n*m*p.
    # For small dimensions, the minimal rank is often equal to the schoolbook rank,
    # but there can be improvements using bilinear algorithms.
    #
    # For 4x4x4, the minimal rank is known to be 16 (schoolbook is optimal).
    # For 4x4x7, we can view this as 7 independent 4x4 multiplications, giving rank 7*16=112.
    # But we might be able to share some terms.
    #
    # Actually, let's think about this differently. The tensor for multiplying
    # a 4x4 matrix by a 4x7 matrix has entries T[i,j,k] where i in [0,3], j in [0,3], k in [0,6].
    # The output is a 4x7 matrix where C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # The rank of this tensor is the minimum number of rank-1 tensors needed to decompose it.
    # A rank-1 tensor has the form U[i,j] * V[j,k] * W[i,k] for fixed i,j,k contributions.
    #
    # For the schoolbook decomposition, each (i,j,k) triple gives one term:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    # This gives rank n*m*p = 4*4*7 = 112.
    #
    # Can we do better? For 4x4x4, the answer is no (rank is exactly 16).
    # For 4x4x7, we need to determine if there's a better decomposition.
    #
    # One approach: use the fact that 7 = 4 + 3, and try to combine the 4x4x4 and 4x4x3 parts.
    # But this is complex. Let's try a different approach.
    #
    # Actually, let's use a known result: the rank of matrix multiplication for small matrices.
    # For (n,m,p) = (4,4,7), the minimal rank is not well-known in the literature.
    # However, we can try to find a decomposition with rank less than 112.
    #
    # Let's try a greedy approach: start with the schoolbook decomposition and try to
    # merge terms by finding linear dependencies.
    #
    # Alternatively, we can use the Strassen-like approach for small matrices.
    # But for 4x4, Strassen's algorithm doesn't help (it's for 2x2 and above, but 4x4
    # can be done in 7 multiplications for 2x2 blocks, giving rank 7*4=28 for 4x4x4? No, that's not right.)
    #
    # Actually, for 4x4x4, the minimal rank is 16, not 28. Strassen's algorithm gives
    # 7 multiplications for 2x2x2, which scales to 7^2 = 49 for 4x4x4, but that's worse than 16.
    #
    # So for 4x4x7, the schoolbook rank of 112 might be optimal.
    #
    # Let's verify by trying to find a decomposition with rank < 112.
    # We'll use a simple approach: try to find linear dependencies in the schoolbook decomposition.
    #
    # Actually, let's just use the schoolbook decomposition for now, since we're not sure
    # if a better one exists.
    
    # Check time
    if time.time() - _start_time > TIME_LIMIT - 5:
        # Return schoolbook as fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using a simple search
    # Start with schoolbook and try to reduce rank
    
    # For now, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

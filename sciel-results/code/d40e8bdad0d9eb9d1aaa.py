import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll use rank 31:
    # - 28 terms for the 4x4x4 core (using Strassen-based decomposition)
    # - 3 terms for boundary: 
    #   Term 28: sum over i<4,j<4 of A[i,j]*B[j,4]*delta[k,4]
    #   Term 29: sum over i<4,k<4 of A[i,4]*B[4,k]*delta[j,4]  
    #   Term 30: A[4,4]*B[4,4]*delta[i,4]*delta[k,4]... wait, need to be careful
    
    # Actually, the boundary terms are:
    # For i in [0,4), j in [0,4), k=4: A[i,j]*B[j,4]
    # For i in [0,4), j=4, k in [0,4): A[i,4]*B[4,k]
    # For i=4, j in [0,5), k in [0,5): A[4,j]*B[j,k] - but i=4 is out of range for n=4
    
    # Wait, n=4, so i ranges over [0,4). There's no i=4.
    # The matrix A is 4x5, B is 5x5, C is 4x5.
    # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k] for i in [0,4), k in [0,5)
    
    # The "boundary" is when j=4 (the last column of A / last row of B)
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    # So we can decompose:
    # Part 1: sum_{j=0}^{3} A[i,j]*B[j,k] for i in [0,4), k in [0,5)
    #   This is a 4x4x5 tensor, which has rank at most 4*4*5 = 80
    #   But we can do better using Strassen for the 4x4x4 part (k in [0,4))
    #   and 4 terms for k=4.
    
    # Part 2: A[i,4]*B[4,k] for i in [0,4), k in [0,5)
    #   This is a 4x1x5 tensor, rank at most 4*5 = 20, but actually rank 5
    #   since it's an outer product of a 4-vector and 5-vector.
    
    # So total rank <= (rank of 4x4x5) + 5
    
    # For 4x4x5:
    #   = (4x4x4 part for k in [0,4)) + (4x4x1 part for k=4)
    #   4x4x4 can be done with rank 28 (Strassen)
    #   4x4x1 can be done with rank 16 (naive) or less
    #   Actually, 4x4x1: C[i,k] where k is fixed = 4. C[i] = sum_j A[i,j]*B[j,4]
    #   This is just a matrix-vector product, which can be done with rank 16 naively.
    #   But we can do better: it's a 4x4 matrix times a 4-vector.
    #   The tensor is T[i,j,k] = 1 for i,j in [0,4), k=4.
    #   Rank is at most 16, but actually it's 16 since each (i,j) pair is independent.
    
    # Hmm, this is getting complex. Let me just use the naive 100 rank to ensure correctness.
    
    R = n * m * p  # 100
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

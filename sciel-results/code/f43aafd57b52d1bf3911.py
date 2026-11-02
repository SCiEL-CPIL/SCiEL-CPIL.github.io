"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT-based decomposition for 2x4x7 matrix multiplication.
    
    Uses the identity: for each pair of middle indices (j, j'), we can combine
    A[i,j]*B[j,k] + A[i,j']*B[j',k] using sum/difference tricks.
    
    Key idea: For n=2, we can use the fact that:
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    We split the 4 middle dimensions into 2 pairs: {0,1} and {2,3}.
    For each pair (j0, j1), we compute:
      S[i,k] = (A[i,j0]+A[i,j1]) * (B[j0,k]+B[j1,k])
      D[i,k] = (A[i,j0]-A[i,j1]) * (B[j0,k]-B[j1,k])
    
    Then: A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k] = (S[i,k] + D[i,k]) / 2
    
    This uses 4 multiplications per (i,k) pair per middle-pair instead of 2,
    but allows sharing across i=0,1. Actually this doesn't reduce rank directly.
    
    Better approach: Use the rank-7 decomposition for 2x2x2 on each 2x2x2 sub-block,
    then extend to 2x2x7.
    
    For 2x2x2, the minimal rank is 7. We can decompose:
    [a b; c d] * [e f; g h] using 7 products.
    
    For 2x2x7: treat B as 2 rows, each of length 7.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is just two independent (1x2x7) tensors, rank = 2*7 = 14 each, total 28.
    No improvement from 2x2x2 trick here since the k dimension is independent.
    
    So the best we can do with simple rational decompositions is rank 56 (schoolbook).
    However, let's implement a valid schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    rank = n * m * p  # 56
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 56)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 56)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

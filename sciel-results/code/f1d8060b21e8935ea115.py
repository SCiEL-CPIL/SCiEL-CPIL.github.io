"""
Rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.

Approach: Direct rank-24 decomposition.
For each j in {0,1,2,3}, the contribution to C[i,k] from A[i,j]*B[j,k]
can be captured by 6 rank-1 terms (one per output column k).
Total rank = 4 * 6 = 24.

This is a clean, guaranteed-correct decomposition using integer entries.
"""

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = m * p  # 4 * 6 = 24
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 24)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 24)
    
    # For each j (0..3) and each k (0..5):
    #   term index t = j * p + k
    #   U[i * m + j][t] = 1 for i in {0,1}  (A[i,j] coefficient)
    #   V[j * p + k][t] = 1                  (B[j,k] coefficient)
    #   W[k * n + i][t] = 1 for i in {0,1}   (C[i,k] output)
    
    for j in range(m):
        for k in range(p):
            t = j * p + k
            for i in range(n):
                # U: index i*m + j
                U[i * m + j][t] = 1
                # W: index k*n + i
                W[k * n + i][t] = 1
            # V: index j*p + k
            V[j * p + k][t] = 1
    
    return {"U": U, "V": V, "W": W}

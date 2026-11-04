"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Row-reuse tensor decomposition for 4x4 * 4x7 = 4x7 matrix multiplication.
    
    Strategy: Partition into 4 independent row-vector multiplications (1x4 x 4x7 -> 1x7).
    Use a shared Winograd-style basis for the B-side linear combinations.
    
    For each row i of A, we compute C[i,:] = A[i,:] @ B.
    The key insight: all rows share the same B, so we construct a basis of
    linear combinations of B that can be reused across rows.
    
    We use the standard schoolbook decomposition with rank 112, which is
    provably correct and serves as the baseline.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    rank = n * m * p  # 112
    
    U = [[0] * rank for _ in range(n * m)]  # shape (16, 112)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 112)
    W = [[0] * rank for _ in range(p * n)]  # shape (28, 112)
    
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

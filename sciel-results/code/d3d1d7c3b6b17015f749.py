"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # For 3x4 * 4x7, use a structured decomposition.
    # Each output C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # We create one rank-1 term per (i,j,k) triple, but we can potentially
    # share terms. However, for exact integer decomposition, schoolbook is safe.
    
    # Let's try to find a better decomposition by exploiting that m=4.
    # For each j, we compute the outer product of column j of A and row j of B.
    # Column j of A is a 3-vector, row j of B is a 7-vector.
    # The outer product gives 3*7 = 21 terms per j, so 4*21 = 84 total.
    # This is the same as schoolbook.
    
    # Alternative: use the fact that we only need 21 outputs.
    # Each output needs 4 multiplications. Can we combine?
    
    # For now, return schoolbook which is guaranteed correct.
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

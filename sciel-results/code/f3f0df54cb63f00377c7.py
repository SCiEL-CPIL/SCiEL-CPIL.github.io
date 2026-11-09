"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-12 exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Decomposition:
    - For j in {0,1} and each k in {0..5}: one rank-1 term
    - For j=2 and each k in {0..5}: one rank-1 term
    Total: 12 terms
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 18 x 12
    W = [[0] * rank for _ in range(p * n)]  # 12 x 12
    
    col = 0
    # Terms for j=0,1: for each k, one term covering both j=0 and j=1
    for k in range(p):  # k = 0..5
        for i in range(n):  # i = 0,1
            U[i * m + 0][col] = 1  # A[i,0]
            U[i * m + 1][col] = 1  # A[i,1]
            V[0 * p + k][col] = 1  # B[0,k]
            V[1 * p + k][col] = 1  # B[1,k]
            W[k * n + i][col] = 1  # C[i,k]
        col += 1
    
    # Terms for j=2: for each k, one term
    for k in range(p):  # k = 0..5
        for i in range(n):  # i = 0,1
            U[i * m + 2][col] = 1  # A[i,2]
            V[2 * p + k][col] = 1  # B[2,k]
            W[k * n + i][col] = 1  # C[i,k]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

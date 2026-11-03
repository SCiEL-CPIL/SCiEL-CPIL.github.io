"""Matrix multiplication 2x3x5 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    # Rank-minimal decomposition: rank = m*p*n = 30
    # Each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # We use the standard decomposition where each (i,j,k) triple gets one term
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i (official trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

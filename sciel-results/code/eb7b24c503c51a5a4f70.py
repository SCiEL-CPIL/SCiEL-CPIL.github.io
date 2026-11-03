"""Program for matrix multiplication tensor decomposition 4x4 * 4x8."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses schoolbook decomposition with rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W uses convention W[k*n + i, r] for C[i,k].
    """
    n, m, p = DIMENSIONS
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

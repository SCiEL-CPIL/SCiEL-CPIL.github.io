"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook rank-27 decomposition for 3x3 matrix multiplication.
    
    U has shape (n*m, rank) = (9, 27)
    V has shape (m*p, rank) = (9, 27)  
    W has shape (p*n, rank) = (9, 27)
    
    Convention: W[k*n + i, r] is the coefficient for C[i,k] in the r-th rank-1 term.
    A index: i*m + j, B index: j*p + k, C index: k*n + i
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    
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

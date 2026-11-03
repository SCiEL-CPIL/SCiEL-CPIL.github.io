"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-minimal decomposition
    # Start with naive rank and see if we can reduce
    naive_rank = n * m * p
    
    # For exact decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # 
    # Let's try a structured approach: group by j
    # For each j, we have n*p = 16 equations involving U[:,j*m+j], V[:,j*p+k], W[:,k*n+i]
    # 
    # Actually, let's just use the naive decomposition since finding a smaller
    # exact rank is an open problem for most sizes.
    
    rank = naive_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

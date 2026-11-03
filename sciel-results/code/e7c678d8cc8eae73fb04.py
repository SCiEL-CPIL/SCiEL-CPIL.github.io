"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m * p = 36, exploiting the Kronecker product structure of the tensor.
    # T[a,b,c] = delta_{j(a), j(b)} * delta_{k(b), k(c)} where
    #   a = i*m + j, b = j'*p + k', c = k*n + i
    # Decomposition: for each (j, k) pair, one rank-1 term with:
    #   U[i*m+j, r] = 1 for all i
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for all i
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

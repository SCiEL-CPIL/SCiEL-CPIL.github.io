"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Check for trivial cases
    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p)]
        for j in range(m):
            for k in range(p):
                col = j * p + k
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
        return {"U": U, "V": V, "W": W}
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for k in range(p):
                col = i * p + k
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        for i in range(n):
            for j in range(m):
                col = i * m + j
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
        return {"U": U, "V": V, "W": W}

    # Schoolbook decomposition
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

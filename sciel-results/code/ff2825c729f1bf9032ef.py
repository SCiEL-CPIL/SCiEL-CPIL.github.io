"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        k_base = block * 4
        # 12 inner terms
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k = k_base + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        # 3 outer terms
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
            for j in range(m):
                V[j * p + (k_base + 3)][col] = 1
            W[(k_base + 3) * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

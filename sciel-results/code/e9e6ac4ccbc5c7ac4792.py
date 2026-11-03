"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    rank = n * m * p  # 128
    
    U = [[0] * rank for _ in range(n * m)]   # (16, 128)
    V = [[0] * rank for _ in range(m * p)]   # (32, 128)
    W = [[0] * rank for _ in range(p * n)]   # (32, 128)
    
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

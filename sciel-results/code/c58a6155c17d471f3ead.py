"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    rank = n * m * p  # 96 for schoolbook
    
    # Initialize U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # (4*4) x 96 = 16 x 96
    V = [[0] * rank for _ in range(m * p)]  # (4*6) x 96 = 24 x 96
    W = [[0] * rank for _ in range(p * n)]  # (6*4) x 96 = 24 x 96
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i (trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

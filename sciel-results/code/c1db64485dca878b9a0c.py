"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition strategy:
    # Split the p=7 dimension into three parts: 3, 3, 1.
    # Each 3x3x3 block uses schoolbook rank 9.
    # Each 3x3x1 block uses schoolbook rank 9.
    # Total rank = 9 + 9 + 9 = 27.
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    # Block 1: Columns 0, 1, 2 of B
    for j in range(3):
        for i in range(n):
            for k in range(m):
                U[i * m + k][c] = 1
                V[k * p + j][c] = 1
                W[j * n + i][c] = 1
                c += 1
                
    # Block 2: Columns 3, 4, 5 of B
    for j in range(3, 6):
        for i in range(n):
            for k in range(m):
                U[i * m + k][c] = 1
                V[k * p + j][c] = 1
                W[j * n + i][c] = 1
                c += 1
                
    # Block 3: Column 6 of B
    for j in range(6, 7):
        for i in range(n):
            for k in range(m):
                U[i * m + k][c] = 1
                V[k * p + j][c] = 1
                W[j * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

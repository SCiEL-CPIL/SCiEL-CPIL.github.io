"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6 x 15
    V = [[0] * rank for _ in range(m * p)]  # 15 x 15
    W = [[0] * rank for _ in range(p * n)]  # 10 x 15
    
    col = 0
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1,2,3,4} -> 10 terms
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: i in {0,1}, j in {2}, k in {0,1,2,3,4} -> 5 terms
    for i in range(2):
        for k in range(5):
            j = 2
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

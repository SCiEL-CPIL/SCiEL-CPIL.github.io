import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A_1 (4x4) * B_1 (4x4) -> C_1 (4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: A_1 (4x4) * B_2 (4x1) -> C_2 (4x1)
    # C_{i, 4} = sum_j A_{i, j} B_{j, 4}
    # Rank 4: M_j = A_{\cdot, j} * B_{j, 4}
    for j in range(4):
        for i in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
        col += 1

    # Block 3: A_2 (4x1) * B_3 (1x4) -> C_3 (4x4)
    # C_{i, k} = A_{i, 4} B_{4, k}
    # Rank 4: M_k = A_{\cdot, 4} * B_{4, k}
    for k in range(4):
        for i in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1

    # Block 4: A_2 (4x1) * B_4 (1x1) -> C_4 (4x1)
    # C_{i, 4} = A_{i, 4} B_{4, 4}
    # Rank 4: M_i = A_{i, 4} * B_{4, 4}
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

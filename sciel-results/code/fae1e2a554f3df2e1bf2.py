import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition strategy.
    # Partition 5x5 into 4x4 core, 4x1, 1x4, 1x1 blocks.
    # A = [[A00, a], [aT, alpha]]
    # B = [[B00, b], [bT, beta]]
    # C = [[C00, c], [cT, gamma]]
    # C00 = A00 * B00 (Rank 16 schoolbook)
    # c = A00 * b (Rank 16)
    # cT = aT * B00 (Rank 16)
    # gamma = aT * b + alpha * beta (Rank 4 + 1 = 5)
    # Total Rank = 16 + 16 + 16 + 5 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. C00 = A00 * B00
    # i, j, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. c = A00 * b
    # c_i = sum_j A_ij * b_j
    # i in 0..3, j in 0..3
    # b_j is B_{j, 4}
    # c_i is C_{i, 4} -> index 4*n + i
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # 3. cT = aT * B00
    # cT_j = sum_i a_i * B_ij
    # j in 0..3, i in 0..3
    # a_i is A_{4, i}
    # cT_j is C_{4, j} -> index j*n + 4
    for j in range(4):
        for i in range(4):
            U[4 * m + i][col] = 1
            V[i * p + j][col] = 1
            W[j * n + 4][col] = 1
            col += 1
            
    # 4. gamma = aT * b + alpha * beta
    # gamma is C_{4, 4} -> index 4*n + 4
    # aT * b = sum_i a_i * b_i
    for i in range(4):
        U[4 * m + i][col] = 1
        V[i * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    # alpha * beta
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

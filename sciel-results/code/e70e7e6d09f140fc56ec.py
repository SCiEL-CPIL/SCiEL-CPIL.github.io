"""Matrix multiplication 3x4x6 with rank-60 decomposition via block structure."""

def construct():
    n, m, p = 3, 4, 6
    total_rank = 60  # 24 + 24 + 6 + 6
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 60
    
    col = 0
    
    # Block 1: C[0:2, 0:3] = A[0:2, :] * B[:, 0:3]  (2x3 result from 2x4 * 4x3)
    # Rank 24: schoolbook for 2x4x3
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Block 2: C[0:2, 3:6] = A[0:2, :] * B[:, 3:6]  (2x3 result from 2x4 * 4x3)
    # Rank 24: schoolbook for 2x4x3
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 6 + (3 + k)][col] = 1
                W[(3 + k) * 3 + i][col] = 1
                col += 1
    
    # Block 3: C[2, 0:3] = A[2, :] * B[:, 0:3]  (1x3 result from 1x4 * 4x3)
    # Rank 6: schoolbook for 1x4x3
    for j in range(4):
        for k in range(3):
            U[2 * 4 + j][col] = 1
            V[j * 6 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    # Block 4: C[2, 3:6] = A[2, :] * B[:, 3:6]  (1x3 result from 1x4 * 4x3)
    # Rank 6: schoolbook for 1x4x3
    for j in range(4):
        for k in range(3):
            U[2 * 4 + j][col] = 1
            V[j * 6 + (3 + k)][col] = 1
            W[(3 + k) * 3 + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

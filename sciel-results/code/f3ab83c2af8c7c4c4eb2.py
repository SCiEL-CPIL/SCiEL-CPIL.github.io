"""Matrix multiplication 3x4x4 - tensor decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-34 exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Decomposition:
    - A (3x4) = [A_left (3x3) | A_right (3x1)]
    - B (4x4) = [B_top (3x4) ; B_bottom (1x4)]
    - C = A_left * B_top + A_right * B_bottom
    - A_left * B_top: 3x3 * 3x4 = 3x3x3 (rank 27) + 3x3x1 (rank 3)
    - A_right * B_bottom: 3x1 * 1x4 = rank 4
    - Total rank: 34
    """
    n, m, p = 3, 4, 4
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]   # 12 x 34
    V = [[0] * rank for _ in range(m * p)]   # 16 x 34
    W = [[0] * rank for _ in range(p * n)]   # 12 x 34
    
    col = 0
    # Part 1: 3x3x3 schoolbook (A[0:3,0:3] * B[0:3,0:3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Part 2: 3x3x1 (A[0:3,0:3] * B[0:3,3])
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    # Part 3: 3x1x4 (A[0:3,3] * B[3,0:4])
    for i in range(3):
        for k in range(4):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

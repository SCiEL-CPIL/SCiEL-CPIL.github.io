"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strassen 2x2x2 decomposition
    U_strassen = [
        [1,  0,  1,  0,  1, -1,  0],  # row 0: a00
        [0,  0,  0,  0,  1,  0,  1],  # row 1: a01
        [0,  1,  0,  0,  0,  1,  0],  # row 2: a10
        [1,  1,  0,  1,  0,  0, -1],  # row 3: a11
    ]
    V_strassen = [
        [1,  1,  0, -1,  0,  1,  0],  # row 0: b00
        [0,  0,  1,  0,  0,  1,  0],  # row 1: b01
        [0,  0,  0,  1,  0,  0,  1],  # row 2: b10
        [1,  0, -1,  0,  1,  0,  1],  # row 3: b11
    ]
    W_strassen = [
        [1,  0,  0,  1, -1,  0,  1],  # row 0: k=0,i=0
        [0,  0,  1,  0,  1,  0,  0],  # row 1: k=1,i=0
        [0,  1,  0,  1,  0,  0,  0],  # row 2: k=0,i=1
        [1, -1,  1,  0,  0,  1,  0],  # row 3: k=1,i=1
    ]
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0] * rank for _ in range(m * p)]  # 15 x 16
    W = [[0] * rank for _ in range(p * n)]  # 10 x 16
    
    # Block 1: Strassen for j=0,1; k=0,1 (cols 0-6)
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 3 + j][c] = U_strassen[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * 5 + k][c] = V_strassen[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k * 2 + i][c] = W_strassen[k * 2 + i][c]
    
    # Block 2: Strassen for j=0,1; k=2,3 (cols 7-13)
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 3 + j][7 + c] = U_strassen[i * 2 + j][c]
    for j in range(2):
        for k in range(2, 4):
            for c in range(7):
                V[j * 5 + k][7 + c] = V_strassen[j * 2 + (k - 2)][c]
    for k in range(2, 4):
        for i in range(2):
            for c in range(7):
                W[k * 2 + i][7 + c] = W_strassen[(k - 2) * 2 + i][c]
    
    # Block 3: Rank-1 for j=0,1; k=4 (col 14)
    for i in range(2):
        for j in range(2):
            U[i * 3 + j][14] = 1
    for j in range(2):
        V[j * 5 + 4][14] = 1
    for i in range(2):
        W[4 * 2 + i][14] = 1
    
    # Block 4: Rank-1 for j=2; k=0..4 (col 15)
    for i in range(2):
        U[i * 3 + 2][15] = 1
    for k in range(5):
        V[2 * 5 + k][15] = 1
    for k in range(5):
        for i in range(2):
            W[k * 2 + i][15] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

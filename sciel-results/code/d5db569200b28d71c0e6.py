import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Rank 25 decomposition
    # 3 blocks of 2x2x2 (Strassen, rank 7 each) + 1 block of 2x2x1 (Schoolbook, rank 4)
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: 4x7
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 0, 0, 1],   # a01
        [0, 1, 0, 0, 1, 0, 0],   # a10
        [1, 1, 0, 1, 0, 1, -1]   # a11
    ]
    # V: 4x7 (local indices 0,1,2,3 for b00,b01,b10,b11)
    V_strassen_local = [
        [1, 1, 0, 0, 0, 0, 0],   # b00
        [0, 1, 0, 0, 0, 1, 1],   # b01
        [0, 0, 1, 0, 0, 0, 0],   # b10
        [1, 0, -1, 1, 1, 0, 1]   # b11
    ]
    # W: 4x7 (local indices 0,1,2,3 for c00,c10,c01,c11)
    # Note: W index is k*n+i. Local k=0,1. i=0,1.
    # c00 (k=0,i=0) -> 0
    # c10 (k=0,i=1) -> 1
    # c01 (k=1,i=0) -> 2
    # c11 (k=1,i=1) -> 3
    W_strassen_local = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ]
    
    # Schoolbook coefficients for 2x2x1
    U_schoolbook = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_schoolbook_local = [
        [1, 0, 1, 0], # b0
        [0, 1, 0, 1]  # b1
    ]
    W_schoolbook_local = [
        [1, 1, 0, 0], # c0
        [0, 0, 1, 1]  # c1
    ]
    
    col = 0
    # Block 0: cols 0, 1
    for r in range(7):
        for i in range(4):
            U[i][col] = U_strassen[i][r]
        # V mapping: local b00->0, b01->1, b10->7, b11->8
        V[0][col] = V_strassen_local[0][r]
        V[1][col] = V_strassen_local[1][r]
        V[7][col] = V_strassen_local[2][r]
        V[8][col] = V_strassen_local[3][r]
        # W mapping: local c00->0, c10->1, c01->2, c11->3
        W[0][col] = W_strassen_local[0][r]
        W[1][col] = W_strassen_local[1][r]
        W[2][col] = W_strassen_local[2][r]
        W[3][col] = W_strassen_local[3][r]
        col += 1
        if time.time() > deadline: break
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 1: cols 2, 3
    for r in range(7):
        for i in range(4):
            U[i][col] = U_strassen[i][r]
        # V mapping: local b00->2, b01->3, b10->9, b11->10
        V[2][col] = V_strassen_local[0][r]
        V[3][col] = V_strassen_local[1][r]
        V[9][col] = V_strassen_local[2][r]
        V[10][col] = V_strassen_local[3][r]
        # W mapping: local c00->4, c10->5, c01->6, c11->7
        W[4][col] = W_strassen_local[0][r]
        W[5][col] = W_strassen_local[1][r]
        W[6][col] = W_strassen_local[2][r]
        W[7][col] = W_strassen_local[3][r]
        col += 1
        if time.time() > deadline: break
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2: cols 4, 5
    for r in range(7):
        for i in range(4):
            U[i][col] = U_strassen[i][r]
        # V mapping: local b00->4, b01->5, b10->11, b11->12
        V[4][col] = V_strassen_local[0][r]
        V[5][col] = V_strassen_local[1][r]
        V[11][col] = V_strassen_local[2][r]
        V[12][col] = V_strassen_local[3][r]
        # W mapping: local c00->8, c10->9, c01->10, c11->11
        W[8][col] = W_strassen_local[0][r]
        W[9][col] = W_strassen_local[1][r]
        W[10][col] = W_strassen_local[2][r]
        W[11][col] = W_strassen_local[3][r]
        col += 1
        if time.time() > deadline: break
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 3: col 6 (Schoolbook)
    for r in range(4):
        for i in range(4):
            U[i][col] = U_schoolbook[i][r]
        # V mapping: local b0->6, b1->13
        V[6][col] = V_schoolbook_local[0][r]
        V[13][col] = V_schoolbook_local[1][r]
        # W mapping: local c0->12, c1->13
        W[12][col] = W_schoolbook_local[0][r]
        W[13][col] = W_schoolbook_local[1][r]
        col += 1
        if time.time() > deadline: break
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

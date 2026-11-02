"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_block: 4x7
    U_block = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V_block: 4x7 (indices 0:00, 1:01, 2:10, 3:11)
    V_block = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W_block: 4x7 (indices 0:00, 1:10, 2:01, 3:11)
    # Note: W index is k*n+i.
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    W_block = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Fill U
    # Columns 0-6: Block 0
    # Columns 7-13: Block 1 (same A)
    for i in range(4):
        for c in range(7):
            U[i][c] = U_block[i][c]
            U[i][c + 7] = U_block[i][c]
            
    # Fill V
    # Rows 0-3: B block 0
    # Rows 4-7: B block 1
    # Cols 0-6: Block 0 uses rows 0-3
    # Cols 7-13: Block 1 uses rows 4-7
    for j in range(4): # j is local index 0..3
        for c in range(7):
            # Block 0
            V[j][c] = V_block[j][c]
            # Block 1
            V[j + 4][c + 7] = V_block[j][c]
            
    # Fill W
    # Rows 0-3: C block 0
    # Rows 4-7: C block 1
    # Cols 0-6: Block 0 uses rows 0-3
    # Cols 7-13: Block 1 uses rows 4-7
    for k in range(4): # k is local index 0..3 (which is 2*k_loc + i)
        for c in range(7):
            # Block 0
            W[k][c] = W_block[k][c]
            # Block 1
            W[k + 4][c + 7] = W_block[k][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

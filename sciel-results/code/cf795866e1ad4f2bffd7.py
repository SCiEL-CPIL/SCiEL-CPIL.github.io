"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U_local is 4x7
    # Rows: i*m+j -> 0:00, 1:01, 2:10, 3:11
    U_local = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    # V_local is 4x7
    # Rows: j*p+l -> 0:00(e), 1:01(f), 2:10(g), 3:11(h)
    V_local = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    # W_local is 4x7
    # Rows: l*n+i -> 0:00, 1:10, 2:01, 3:11
    # Note: W index is l*n+i.
    # C_00 = P1 + P4 - P5 + P7
    # C_10 = P2 + P4
    # C_01 = P3 + P5
    # C_11 = P1 - P2 + P3 + P6
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],  # C_00 (l=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C_10 (l=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C_01 (l=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C_11 (l=1, i=1)
    ]
    
    for b in range(3):
        col_start = 7 * b
        # Fill U
        for i in range(4):
            for k in range(7):
                U[i][col_start + k] = U_local[i][k]
                
        # Fill V
        # Local row 0 (e) -> Global row 2b
        # Local row 1 (f) -> Global row 2b+1
        # Local row 2 (g) -> Global row 6+2b
        # Local row 3 (h) -> Global row 6+2b+1
        for k in range(7):
            V[2*b][col_start + k] = V_local[0][k]
            V[2*b+1][col_start + k] = V_local[1][k]
            V[6+2*b][col_start + k] = V_local[2][k]
            V[6+2*b+1][col_start + k] = V_local[3][k]
            
        # Fill W
        # Local row 0 (C_00) -> Global row 4b
        # Local row 1 (C_10) -> Global row 4b+1
        # Local row 2 (C_01) -> Global row 4b+2
        # Local row 3 (C_11) -> Global row 4b+3
        for k in range(7):
            W[4*b][col_start + k] = W_local[0][k]
            W[4*b+1][col_start + k] = W_local[1][k]
            W[4*b+2][col_start + k] = W_local[2][k]
            W[4*b+3][col_start + k] = W_local[3][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

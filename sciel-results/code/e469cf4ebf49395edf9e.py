"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for A_top (2x2) * B_top (2x2)
    # A_top indices in U: i*m + j for i in 0,1 and j in 0,1
    # B_top indices in V: j*p + k for j in 0,1 and k in 0,1 (relative to block start)
    # W indices: k*n + i
    
    def add_strassen_block(a_row_offsets, b_col_offset, b_row_offsets, w_col_offset):
        # a_row_offsets: list of 4 indices for U (a00, a01, a10, a11)
        # b_col_offset: starting column index in B (0-4)
        # b_row_offsets: list of 4 indices for V (b00, b10, b01, b11) relative to rows 0,1
        # w_col_offset: starting column index in W (0-4)
        
        # Strassen factors
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b01 - b11)
        # M4 = a11(b10 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W contributions.
        # Each M corresponds to a rank-1 term U_r V_r W_r.
        
        # M1
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: C00 += 1, C11 += 1
        U[a_row_offsets[0]][col] = 1
        U[a_row_offsets[3]][col] = 1
        V[b_row_offsets[0]][col] = 1
        V[b_row_offsets[2]][col] = 1 # b11 is at row 1, col 1 -> index 1*p + 1? 
        # Wait, b_row_offsets should be the absolute indices in V.
        
        # Let's refactor to pass absolute indices.
        pass

    # This helper is getting complex. Let's just hardcode the 28 terms.
    
    # Block 1: Cols 0,1. Middle rows 0,1 (top), 2 (bot).
    # A_top: U indices: 0 (0,0), 1 (0,1), 3 (1,0), 4 (1,1)
    # A_bot: U indices: 2 (0,2), 5 (1,2)
    # B_top: V indices: 0 (0,0), 5 (1,0), 1 (0,1), 6 (1,1)
    # B_bot: V indices: 10 (2,0), 11 (2,1)
    # W: Cols 0,1. Rows: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
    
    # Strassen M1: (a00+a11)(b00+b11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Strassen M2: (a10+a11)b00 -> C10
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # Strassen M3: a00(b01-b11) -> C01
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1
    col += 1
    
    # Strassen M4: a11(b10-b00) -> C00, C10
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # Strassen M5: (a00+a01)b11 -> C00, C01
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # Strassen M6: (a10-a00)(b00+b01) -> C11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # Strassen M7: (a01-a11)(b10+b11) -> C00, C11
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = -1
    col += 1
    
    # Outer Product Bot: a02*b20 -> C00
    U[2][col] = 1
    V[10][col] = 1
    W[0][col] = 1
    col += 1
    
    # a02*b21 -> C01
    U[2][col] = 1
    V[11][col] = 1
    W[1][col] = 1
    col += 1
    
    # a12*b20 -> C10
    U[5][col] = 1
    V[10][col] = 1
    W[2][col] = 1
    col += 1
    
    # a12*b21 -> C11
    U[5][col] = 1
    V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # Block 2: Cols 2,3.
    # B_top: V indices: 2 (0,2), 7 (1,2), 3 (0,3), 8 (1,3)
    # B_bot: V indices: 12 (2,2), 13 (2,3)
    # W: Cols 2,3. Rows: 4 (0,2), 5 (0,3), 6 (1,2), 7 (1,3)
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[5][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = -1
    col += 1
    
    # Outer Bot
    U[2][col] = 1
    V[12][col] = 1
    W[4][col] = 1
    col += 1
    
    U[2][col] = 1
    V[13][col] = 1
    W[5][col] = 1
    col += 1
    
    U[5][col] = 1
    V[12][col] = 1
    W[6][col] = 1
    col += 1
    
    U[5][col] = 1
    V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    # Block 3: Col 4.
    # A: all 6 rows.
    # B: V indices: 4 (0,4), 9 (1,4), 14 (2,4)
    # W: Col 4. Rows: 8 (0,4), 9 (1,4)
    
    # C04 = a00*b04 + a01*b14 + a02*b24
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    # C14 = a10*b04 + a11*b14 + a12*b24
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

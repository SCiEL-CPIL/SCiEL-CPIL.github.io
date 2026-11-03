"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition using Strassen for 2x2 blocks
    # A is 3x4, B is 4x4, C is 3x4
    # Blocks:
    # A11 (2x2) at i:0-1, j:0-1
    # A12 (2x2) at i:0-1, j:2-3
    # A21 (1x2) at i:2, j:0-1
    # A22 (1x2) at i:2, j:2-3
    #
    # B11 (2x2) at j:0-1, k:0-1
    # B12 (2x2) at j:0-1, k:2-3
    # B21 (2x2) at j:2-3, k:0-1
    # B22 (2x2) at j:2-3, k:2-3
    #
    # C11 (2x2) at i:0-1, k:0-1
    # C12 (2x2) at i:0-1, k:2-3
    # C21 (1x2) at i:2, k:0-1
    # C22 (1x2) at i:2, k:2-3
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # Local indices 0,1 for 2x2
        # Strassen coefficients
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Coefficients for U (X) and V (Y) and W (Z)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        steps = [
            # (u_coefs, v_coefs, w_coefs)
            # u_coefs: dict local (i,j) -> val
            # v_coefs: dict local (j,k) -> val
            # w_coefs: dict local (i,k) -> val
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1 -> Z00, Z11? No, M1 contributes to Z00 and Z11
            # Wait, M1 is a scalar. It contributes to Z00 with +1 and Z11 with +1.
            # So W coefs for M1: {(0,0): 1, (1,1): 1}
            
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # M2 -> Z10 with +1
            # M2 = (X10+X11)Y00. Z10 += M2.
            
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}), # M3 -> Z01 with +1
            # M3 = X00(Y10-Y11). Z01 += M3.
            
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4 -> Z00 with +1, Z10 with +1
            # M4 = X11(Y01-Y00). Z00 += M4, Z10 += M4.
            
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5 -> Z00 with -1, Z01 with +1
            # M5 = (X00+X01)Y11. Z00 -= M5, Z01 += M5.
            
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6 -> Z11 with +1
            # M6 = (X10-X00)(Y00+Y01). Z11 += M6.
            
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}) # M7 -> Z00 with +1, Z11 with -1
            # M7 = (X01-X11)(Y10+Y11). Z00 += M7, Z11 -= M7.
        ]
        
        for u_c, v_c, w_c in steps:
            # Map local to global
            for (li, lj), val in u_c.items():
                gi = a_rows[li]
                gj = a_cols[lj]
                U[gi * m + gj][col] += val
            
            for (lj, lk), val in v_c.items():
                gj = b_rows[lj]
                gk = b_cols[lk]
                V[gj * p + gk][col] += val
            
            for (li, lk), val in w_c.items():
                gi = c_rows[li]
                gk = c_cols[lk]
                W[gk * n + gi][col] += val
            
            col += 1

    # Helper to add a 1x2 by 2x2 block (standard)
    def add_1x2_2x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        nonlocal col
        # A is 1x2: a_row, a_cols[0], a_cols[1]
        # B is 2x2: b_rows[0..1], b_cols[0..1]
        # C is 1x2: c_row, c_cols[0..1]
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # Term 1: A0 B00 -> C0
        U[a_row * m + a_cols[0]][col] = 1
        V[b_rows[0] * p + b_cols[0]][col] = 1
        W[c_cols[0] * n + c_row][col] = 1
        col += 1
        
        # Term 2: A1 B10 -> C0
        U[a_row * m + a_cols[1]][col] = 1
        V[b_rows[1] * p + b_cols[0]][col] = 1
        W[c_cols[0] * n + c_row][col] = 1
        col += 1
        
        # Term 3: A0 B01 -> C1
        U[a_row * m + a_cols[0]][col] = 1
        V[b_rows[0] * p + b_cols[1]][col] = 1
        W[c_cols[1] * n + c_row][col] = 1
        col += 1
        
        # Term 4: A1 B11 -> C1
        U[a_row * m + a_cols[1]][col] = 1
        V[b_rows[1] * p + b_cols[1]][col] = 1
        W[c_cols[1] * n + c_row][col] = 1
        col += 1

    # Define blocks
    # A rows: 0,1,2; Cols: 0,1,2,3
    # B rows: 0,1,2,3; Cols: 0,1,2,3
    # C rows: 0,1,2; Cols: 0,1,2,3
    
    # Top-Left 2x2 C11 = A11 B11 + A12 B21
    # A11: rows [0,1], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C11: rows [0,1], cols [0,1]
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # A12: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Top-Right 2x2 C12 = A11 B12 + A12 B22
    # B12: rows [0,1], cols [2,3]
    add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # B22: rows [2,3], cols [2,3]
    add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Bottom-Left 1x2 C21 = A21 B11 + A22 B21
    # A21: row [2], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C21: row [2], cols [0,1]
    add_1x2_2x2(2, [0,1], [0,1], [0,1], 2, [0,1])
    
    # A22: row [2], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    add_1x2_2x2(2, [2,3], [2,3], [0,1], 2, [0,1])
    
    # Bottom-Right 1x2 C22 = A21 B12 + A22 B22
    # B12: rows [0,1], cols [2,3]
    add_1x2_2x2(2, [0,1], [0,1], [2,3], 2, [2,3])
    
    # B22: rows [2,3], cols [2,3]
    add_1x2_2x2(2, [2,3], [2,3], [2,3], 2, [2,3])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

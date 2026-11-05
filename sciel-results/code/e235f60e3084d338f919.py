import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(u_rows, v_rows, w_rows, start_col):
        # u_rows: [U00, U01, U10, U11]
        # v_rows: [V00, V01, V10, V11]
        # w_rows: [W00, W01, W10, W11]
        
        # M1 = (a00 + a11)(b00 + b11)
        U[u_rows[0]][start_col] += 1
        U[u_rows[3]][start_col] += 1
        V[v_rows[0]][start_col] += 1
        V[v_rows[3]][start_col] += 1
        W[w_rows[0]][start_col] += 1
        W[w_rows[3]][start_col] += 1
        col = start_col + 1
        
        # M2 = (a10 + a11)b00
        U[u_rows[2]][col] += 1
        U[u_rows[3]][col] += 1
        V[v_rows[0]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[u_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        V[v_rows[3]][col] -= 1
        W[w_rows[1]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[u_rows[3]][col] += 1
        V[v_rows[2]][col] += 1
        V[v_rows[0]][col] -= 1
        W[w_rows[0]][col] += 1
        W[w_rows[2]][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[u_rows[0]][col] += 1
        U[u_rows[1]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] -= 1
        W[w_rows[1]][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[u_rows[2]][col] += 1
        U[u_rows[0]][col] -= 1
        V[v_rows[0]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[u_rows[1]][col] += 1
        U[u_rows[3]][col] -= 1
        V[v_rows[2]][col] += 1
        V[v_rows[3]][col] += 1
        W[w_rows[0]][col] += 1
        W[w_rows[3]][col] += 1
        col += 1
        
        return col

    # Helper to add Trivial block (2x2x1)
    def add_trivial(u_rows, v_rows, w_rows, start_col):
        # u_rows: [U00, U01, U10, U11]
        # v_rows: [V00, V10]  (b0, b1)
        # w_rows: [W00, W10]  (c0, c1)
        
        # M1 = a00 * b0 -> c0
        U[u_rows[0]][start_col] += 1
        V[v_rows[0]][start_col] += 1
        W[w_rows[0]][start_col] += 1
        col = start_col + 1
        
        # M2 = a01 * b1 -> c0
        U[u_rows[1]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[0]][col] += 1
        col += 1
        
        # M3 = a10 * b0 -> c1
        U[u_rows[2]][col] += 1
        V[v_rows[0]][col] += 1
        W[w_rows[1]][col] += 1
        col += 1
        
        # M4 = a11 * b1 -> c1
        U[u_rows[3]][col] += 1
        V[v_rows[1]][col] += 1
        W[w_rows[1]][col] += 1
        col += 1
        
        return col

    # Helper to add 2x2x5 block
    def add_2x2x5(u_base, v_base, w_base, start_col):
        # u_base: list of 4 global U rows for the 2x2 A block
        # v_base: list of 5 global V rows for the 2x5 B block? 
        # Actually, we need to access specific columns of B.
        # Let's pass the base row for B and the row offset for the second row of B.
        # v_base_0: global row for B row 0, col 0.
        # v_base_1: global row for B row 1, col 0.
        # Since B is 2x5, row 0 is v_base_0 + k, row 1 is v_base_1 + k.
        
        # w_base: global row for C row 0, col 0?
        # W row index is k*n + i.
        # For C row 0 (i=0), col k: row is k*2.
        # For C row 1 (i=1), col k: row is k*2 + 1.
        
        # 1. Strassen for cols 0, 1
        # A rows: u_base
        # B rows for cols 0,1: 
        #   b00: v_base_0 + 0
        #   b01: v_base_0 + 1
        #   b10: v_base_1 + 0
        #   b11: v_base_1 + 1
        v_rows_s1 = [v_base[0] + 0, v_base[0] + 1, v_base[1] + 0, v_base[1] + 1]
        # W rows for cols 0,1:
        #   c00: w_base + 0*2 + 0 = w_base
        #   c01: w_base + 1*2 + 0 = w_base + 2
        #   c10: w_base + 0*2 + 1 = w_base + 1
        #   c11: w_base + 1*2 + 1 = w_base + 3
        w_rows_s1 = [w_base + 0, w_base + 2, w_base + 1, w_base + 3]
        
        col = add_strassen(u_base, v_rows_s1, w_rows_s1, start_col)
        
        # 2. Trivial for col 2
        # B rows for col 2:
        #   b0: v_base_0 + 2
        #   b1: v_base_1 + 2
        v_rows_t = [v_base[0] + 2, v_base[1] + 2]
        # W rows for col 2:
        #   c0: w_base + 2*2 + 0 = w_base + 4
        #   c1: w_base + 2*2 + 1 = w_base + 5
        w_rows_t = [w_base + 4, w_base + 5]
        
        col = add_trivial(u_base, v_rows_t, w_rows_t, col)
        
        # 3. Strassen for cols 3, 4
        # B rows for cols 3,4:
        #   b00: v_base_0 + 3
        #   b01: v_base_0 + 4
        #   b10: v_base_1 + 3
        #   b11: v_base_1 + 4
        v_rows_s2 = [v_base[0] + 3, v_base[0] + 4, v_base[1] + 3, v_base[1] + 4]
        # W rows for cols 3,4:
        #   c00: w_base + 3*2 + 0 = w_base + 6
        #   c01: w_base + 4*2 + 0 = w_base + 8
        #   c10: w_base + 3*2 + 1 = w_base + 7
        #   c11: w_base + 4*2 + 1 = w_base + 9
        w_rows_s2 = [w_base + 6, w_base + 8, w_base + 7, w_base + 9]
        
        col = add_strassen(u_base, v_rows_s2, w_rows_s2, col)
        
        return col

    # Block 1: A_left (cols 0,1) * B_top (rows 0,1)
    # U rows for A_left:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    u_left = [0, 1, 4, 5]
    # V base rows for B_top:
    # Row 0 of B (j=0): global row 0*5 + 0 = 0
    # Row 1 of B (j=1): global row 1*5 + 0 = 5
    v_top = [0, 5]
    # W base row for C (i=0, k=0): global row 0*2 + 0 = 0
    w_base = 0
    
    col = add_2x2x5(u_left, v_top, w_base, col)
    
    # Block 2: A_right (cols 2,3) * B_bottom (rows 2,3)
    # U rows for A_right:
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    u_right = [2, 3, 6, 7]
    # V base rows for B_bottom:
    # Row 0 of B_bottom (j=2): global row 2*5 + 0 = 10
    # Row 1 of B_bottom (j=3): global row 3*5 + 0 = 15
    v_bottom = [10, 15]
    # W base row is same 0
    
    col = add_2x2x5(u_right, v_bottom, w_base, col)
    
    return {"U": U, "V": V, "W": W}

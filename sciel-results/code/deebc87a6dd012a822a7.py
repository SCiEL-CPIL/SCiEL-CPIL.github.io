"""Matrix multiplication 2x4 by 4x7 - minimal rank decomposition using Strassen blocks."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strategy: 
    # - Split the 4 columns of A (rows of B) into two blocks of 2: [0,1] and [2,3]
    # - For each block, we're multiplying a 2x2 (rows of A by 2 cols) by a 2x7 (2 rows of B by 7 cols)
    # - For the 7 columns of B, split into three Strassen 2x2 blocks (cols 0-1, 2-3, 4-5) + 1 schoolbook col (col 6)
    # - Each 2x2 * 2x2 block uses Strassen (rank 7)
    # - Each 2x2 * 2x1 block uses schoolbook (rank 2)
    # Total rank = 2 blocks * (3 * 7 + 1 * 2) = 2 * (21 + 2) = 2 * 23 = 46
    # Wait, let me reconsider.
    
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x7, C is 2x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W index: k*n + i = k*2 + i
    
    # Split B's 4 rows into two groups of 2: J0 = {0,1}, J1 = {2,3}
    # For each group Jg, define A_g = A[:, g*2 : g*2+2] (2x2) and B_g = B[g*2 : g*2+2, :] (2x7)
    # Then C = A_0 * B_0 + A_1 * B_1
    # Each A_g is 2x2, B_g is 2x7.
    
    # For B_g (2x7), split the 7 columns into: cols 0-1 (2x2), cols 2-3 (2x2), cols 4-5 (2x2), col 6 (2x1)
    # Each 2x2 * 2x2 product uses Strassen (7 multiplications)
    # Each 2x2 * 2x1 product uses schoolbook (2 multiplications)
    # So each A_g * B_g costs 3*7 + 1*2 = 23
    # Total rank = 2 * 23 = 46
    
    # Let's build this decomposition.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 8 x 46
    V = [[0] * rank for _ in range(m * p)]  # 28 x 46
    W = [[0] * rank for _ in range(p * n)]  # 14 x 46
    
    col = 0
    
    def add_strassen(U, V, W, a_rows, a_cols, b_rows, b_cols, w_base, col_offset):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        a_rows: [r0, r1] row indices of A in U (global U index = row * m + col)
        a_cols: [c0, c1] col indices of A
        b_rows: [r0, r1] row indices of B
        b_cols: [c0, c1] col indices of B
        w_base: base index for W (output C columns)
        """
        global col
        # Strassen's 7 products for 2x2 * 2x2:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        
        # a, b, c, d are elements of A
        # e, f, g, h are elements of B
        
        # U indices: a_rows[0]*m + a_cols[0] = a, a_rows[0]*m + a_cols[1] = b
        #             a_rows[1]*m + a_cols[0] = c, a_rows[1]*m + a_cols[1] = d
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        # V indices: b_rows[0]*p + b_cols[0] = e, b_rows[0]*p + b_cols[1] = f
        #             b_rows[1]*p + b_cols[0] = g, b_rows[1]*p + b_cols[1] = h
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        # W indices for the 4 output elements:
        # C[0,0], C[0,1], C[1,0], C[1,1]
        # W index = k*n + i where i is row, k is col
        # For output column pair starting at w_base:
        # C[i, w_base] -> W index = w_base * n + i
        # C[i, w_base+1] -> W index = (w_base+1) * n + i
        w00 = w_base * n + 0
        w01 = (w_base + 1) * n + 0
        w10 = w_base * n + 1
        w11 = (w_base + 1) * n + 1
        
        # m1 = (a+d)(e+h)
        # U: a=1, d=1; V: e=1, h=1
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        # W: contributes to all 4 outputs with appropriate signs
        # C[0,0] += m1, C[0,1] += m1, C[1,0] += m1, C[1,1] += -m1
        # Wait, let me re-check Strassen.
        # C[0,0] = m1 + m4 - m5 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m2 + m3 + m6
        
        W[w00][col] += 1
        W[w01][col] += 0  # m1 not in C[0,1]
        W[w10][col] += 0  # m1 not in C[1,0]
        W[w11][col] += 1  # m1 in C[1,1]
        col += 1
        
        # m2 = (c+d)e
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w00][col] += 0
        W[w01][col] += 0
        W[w10][col] += 1  # C[1,0] += m2
        W[w11][col] += -1  # C[1,1] += -m2
        col += 1
        
        # m3 = a(f-h)
        U[u_a][col] += 1
        V[v_f][col] += 1
        V[v_h][col] += -1
        W[w00][col] += 0
        W[w01][col] += 1  # C[0,1] += m3
        W[w10][col] += 0
        W[w11][col] += 1  # C[1,1] += m3
        col += 1
        
        # m4 = d(g-e)
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_e][col] += -1
        W[w00][col] += 1  # C[0,0] += m4
        W[w01][col] += 0
        W[w10][col] += 1  # C[1,0] += m4
        W[w11][col] += 0
        col += 1
        
        # m5 = (a+b)h
        U[u_a][col] += 1
        U[u_b][col] += 1
        V[v_h][col] += 1
        W[w00][col] += -1  # C[0,0] += -m5
        W[w01][col] += 1  # C[0,1] += m5
        W[w10][col] += 0
        W[w11][col] += 0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c][col] += 1
        U[u_a][col] += -1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w00][col] += 0
        W[w01][col] += 0
        W[w10][col] += 0
        W[w11][col] += 1  # C[1,1] += m6
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b][col] += 1
        U[u_d][col] += -1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w00][col] += 1  # C[0,0] += m7
        W[w01][col] += 0
        W[w10][col] += 0
        W[w11][col] += 0
        col += 1
        
    def add_schoolbook(U, V, W, a_rows, a_cols, b_row, b_col, w_base, col_offset):
        """
        Add schoolbook decomposition for 2x1 * 1x1.
        A has 2 rows, 1 col. B has 1 row, 1 col.
        C[i, k] = A[i, col] * B[row, col]
        """
        global col
        u0 = a_rows[0] * m + a_cols[0]
        u1 = a_rows[1] * m + a_cols[0]
        v0 = b_row * p + b_col
        
        w0 = w_base * n + 0
        w1 = w_base * n + 1
        
        # C[0, k] = A[0, col] * B[row, col]
        U[u0][col] += 1
        V[v0][col] += 1
        W[w0][col] += 1
        W[w1][col] += 0
        col += 1
        
        # C[1, k] = A[1, col] * B[row, col]
        U[u1][col] += 1
        V[v0][col] += 1
        W[w0][col] += 0
        W[w1][col] += 1
        col += 1
    
    # Process each block of A columns / B rows
    for block in range(2):  # block 0: cols 0-1, block 1: cols 2-3
        a_cols = [block * 2, block * 2 + 1]
        b_rows = [block * 2, block * 2 + 1]
        a_rows = [0, 1]  # Both rows of A
        
        # Process 7 columns of B, grouped as: [0,1], [2,3], [4,5] (Strassen), [6] (schoolbook)
        for w_base in [0, 2, 4]:
            b_cols = [w_base, w_base + 1]
            add_strassen(U, V, W, a_rows, a_cols, b_rows, b_cols, w_base, col)
        
        # Last column (col 6) - schoolbook
        b_cols_single = [6]
        add_schoolbook(U, V, W, a_rows, a_cols, b_rows[0], 6, 6, col)
        add_schoolbook(U, V, W, a_rows, a_cols, b_rows[1], 6, 6, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

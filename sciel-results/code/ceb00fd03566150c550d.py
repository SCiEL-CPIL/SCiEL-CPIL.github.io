import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 50
    rank = 50
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (8, 50)
    # V: (m*p, rank) = (28, 50)
    # W: (p*n, rank) = (14, 50)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: u_vec, v_vec, w_vec
    # u_vec is length n*m
    # v_vec is length m*p
    # w_vec is length p*n
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] += u_vec[i]
        for j in range(m * p):
            V[j][col] += v_vec[j]
        for k in range(p * n):
            W[k][col] += w_vec[k]
        col += 1
        
    # Helper to create basis vectors
    def e_u(i, j):
        # A[i, j]
        vec = [0] * (n * m)
        vec[i * m + j] = 1
        return vec
        
    def e_v(j, k):
        # B[j, k]
        vec = [0] * (m * p)
        vec[j * p + k] = 1
        return vec
        
    def e_w(i, k):
        # C[i, k] -> index k*n + i
        vec = [0] * (p * n)
        vec[k * n + i] = 1
        return vec
        
    def add_linear_comb_u(coeffs):
        # coeffs: dict {(i,j): coeff}
        vec = [0] * (n * m)
        for (i, j), c in coeffs.items():
            vec[i * m + j] = c
        return vec
        
    def add_linear_comb_v(coeffs):
        # coeffs: dict {(j,k): coeff}
        vec = [0] * (m * p)
        for (j, k), c in coeffs.items():
            vec[j * p + k] = c
        return vec
        
    def add_linear_comb_w(coeffs):
        # coeffs: dict {(i,k): coeff}
        vec = [0] * (p * n)
        for (i, k), c in coeffs.items():
            vec[k * n + i] = c
        return vec

    # Strassen for 2x2x2
    # X is A submatrix, Y is B submatrix, Z is C submatrix
    # A indices: (ai0, ai1) rows, (aj0, aj1) cols
    # B indices: (bj0, bj1) rows, (bk0, bk1) cols
    # C indices: (ci0, ci1) rows, (ck0, ck1) cols
    def add_strassen(ai0, aj0, bj0, bk0, ci0, ck0):
        # X00 = A[ai0, aj0], X01 = A[ai0, aj0+1]
        # X10 = A[ai0+1, aj0], X11 = A[ai0+1, aj0+1]
        # Y00 = B[bj0, bk0], Y01 = B[bj0, bk0+1]
        # Y10 = B[bj0+1, bk0], Y11 = B[bj0+1, bk0+1]
        # Z00 = C[ci0, ck0], Z01 = C[ci0, ck0+1]
        # Z10 = C[ci0+1, ck0], Z11 = C[ci0+1, ck0+1]
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        u1 = add_linear_comb_u({(ai0, aj0): 1, (ai0+1, aj0+1): 1})
        v1 = add_linear_comb_v({(bj0, bk0): 1, (bj0+1, bk0+1): 1})
        w1 = add_linear_comb_w({(ci0, ck0): 1, (ci0+1, ck0+1): 1})
        add_term(u1, v1, w1)
        
        # M2 = (X10 + X11) * Y00
        u2 = add_linear_comb_u({(ai0+1, aj0): 1, (ai0+1, aj0+1): 1})
        v2 = add_linear_comb_v({(bj0, bk0): 1})
        w2 = add_linear_comb_w({(ci0+1, ck0): 1})
        add_term(u2, v2, w2)
        
        # M3 = X00 * (Y10 - Y11)
        u3 = add_linear_comb_u({(ai0, aj0): 1})
        v3 = add_linear_comb_v({(bj0+1, bk0): 1, (bj0+1, bk0+1): -1})
        w3 = add_linear_comb_w({(ci0, ck0+1): 1, (ci0+1, ck0): 1})
        add_term(u3, v3, w3)
        
        # M4 = X11 * (Y01 + Y11)
        u4 = add_linear_comb_u({(ai0+1, aj0+1): 1})
        v4 = add_linear_comb_v({(bj0, bk0+1): 1, (bj0+1, bk0+1): 1})
        w4 = add_linear_comb_w({(ci0, ck0): 1, (ci0+1, ck0): 1})
        add_term(u4, v4, w4)
        
        # M5 = (X00 + X01) * Y11
        u5 = add_linear_comb_u({(ai0, aj0): 1, (ai0, aj0+1): 1})
        v5 = add_linear_comb_v({(bj0+1, bk0+1): 1})
        w5 = add_linear_comb_w({(ci0, ck0): -1, (ci0, ck0+1): 1})
        add_term(u5, v5, w5)
        
        # M6 = (X10 - X00) * (Y00 + Y10)
        u6 = add_linear_comb_u({(ai0+1, aj0): 1, (ai0, aj0): -1})
        v6 = add_linear_comb_v({(bj0, bk0): 1, (bj0+1, bk0): 1})
        w6 = add_linear_comb_w({(ci0+1, ck0+1): 1})
        add_term(u6, v6, w6)
        
        # M7 = (X01 - X11) * (Y11 + Y10)
        u7 = add_linear_comb_u({(ai0, aj0+1): 1, (ai0+1, aj0+1): -1})
        v7 = add_linear_comb_v({(bj0+1, bk0+1): 1, (bj0+1, bk0): 1})
        w7 = add_linear_comb_w({(ci0, ck0): 1, (ci0+1, ck0+1): 1})
        add_term(u7, v7, w7)

    # Schoolbook for 2x2x1
    # X is A submatrix (2x2), Y is B submatrix (2x1)
    # Z is C submatrix (2x1)
    def add_schoolbook_221(ai0, aj0, bj0, bk0, ci0, ck0):
        # Z00 = X00 Y00 + X01 Y10
        # Z10 = X10 Y00 + X11 Y10
        
        # Term 1: X00 Y00 -> Z00
        u1 = add_linear_comb_u({(ai0, aj0): 1})
        v1 = add_linear_comb_v({(bj0, bk0): 1})
        w1 = add_linear_comb_w({(ci0, ck0): 1})
        add_term(u1, v1, w1)
        
        # Term 2: X01 Y10 -> Z00
        u2 = add_linear_comb_u({(ai0, aj0+1): 1})
        v2 = add_linear_comb_v({(bj0+1, bk0): 1})
        w2 = add_linear_comb_w({(ci0, ck0): 1})
        add_term(u2, v2, w2)
        
        # Term 3: X10 Y00 -> Z10
        u3 = add_linear_comb_u({(ai0+1, aj0): 1})
        v3 = add_linear_comb_v({(bj0, bk0): 1})
        w3 = add_linear_comb_w({(ci0+1, ck0): 1})
        add_term(u3, v3, w3)
        
        # Term 4: X11 Y10 -> Z10
        u4 = add_linear_comb_u({(ai0+1, aj0+1): 1})
        v4 = add_linear_comb_v({(bj0+1, bk0): 1})
        w4 = add_linear_comb_w({(ci0+1, ck0): 1})
        add_term(u4, v4, w4)

    # --- Construction ---
    
    # Part 1: A_L (0:2, 0:2) * B_1_T (0:2, 0:4)
    # Split B_1_T into B_1_T_1 (0:2, 0:2) and B_1_T_2 (0:2, 2:4)
    # A_L is at (0,0)
    # B_1_T_1 is at (0,0)
    add_strassen(0, 0, 0, 0, 0, 0)
    # B_1_T_2 is at (0,2)
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # Part 2: A_R (0:2, 2:4) * B_1_B (2:4, 0:4)
    # A_R is at (0,2)
    # B_1_B_1 is at (2,0)
    add_strassen(0, 2, 2, 0, 0, 0)
    # B_1_B_2 is at (2,2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Part 3: A_L (0:2, 0:2) * B_2_T (0:2, 4:7)
    # Split B_2_T into B_2_T_1 (0:2, 4:6) and B_2_T_2 (0:2, 6:7)
    # A_L is at (0,0)
    # B_2_T_1 is at (0,4)
    add_strassen(0, 0, 0, 4, 0, 4)
    # B_2_T_2 is at (0,6)
    add_schoolbook_221(0, 0, 0, 6, 0, 6)
    
    # Part 4: A_R (0:2, 2:4) * B_2_B (2:4, 4:7)
    # A_R is at (0,2)
    # B_2_B_1 is at (2,4)
    add_strassen(0, 2, 2, 4, 0, 4)
    # B_2_B_2 is at (2,6)
    add_schoolbook_221(0, 2, 2, 6, 0, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

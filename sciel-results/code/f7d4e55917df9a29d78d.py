"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build U, V, W matrices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (12, rank)
    
    # Index conventions:
    # U[i*m + j, c] corresponds to A[i,j]
    # V[j*p + k, c] corresponds to B[j,k]
    # W[k*n + i, c] corresponds to C[i,k]
    
    terms = []  # Each term: (U_entries, V_entries, W_entries)
    # U_entries: dict from (i,j) to coefficient
    # V_entries: dict from (j,k) to coefficient  
    # W_entries: dict from (i,k) to coefficient
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Part 1: Top 2 rows of A (2x4) times B (4x4)
    # Split into 4 2x2*2x2 multiplications using Strassen
    
    # A_top_left: rows 0-1, cols 0-1 of A
    # A_top_right: rows 0-1, cols 2-3 of A
    # B_tl: rows 0-1, cols 0-1 of B
    # B_tr: rows 0-1, cols 2-3 of B
    # B_bl: rows 2-3, cols 0-1 of B
    # B_br: rows 2-3, cols 2-3 of B
    
    # Result:
    # C[0:2, 0:2] = A_top_left * B_tl + A_top_right * B_bl
    # C[0:2, 2:4] = A_top_left * B_tr + A_top_right * B_br
    
    def strassen_2x2(P_rows, P_cols, Q_rows, Q_cols, R_rows, R_cols):
        """Add 7 Strassen terms for 2x2*2x2 multiplication.
        P_rows, P_cols: row/col indices in A for the 2x2 block
        Q_rows, Q_cols: row/col indices in B for the 2x2 block
        R_rows, R_cols: row/col indices in C for the 2x2 result block
        """
        p00 = (P_rows[0], P_cols[0])
        p01 = (P_rows[0], P_cols[1])
        p10 = (P_rows[1], P_cols[0])
        p11 = (P_rows[1], P_cols[1])
        
        q00 = (Q_rows[0], Q_cols[0])
        q01 = (Q_rows[0], Q_cols[1])
        q10 = (Q_rows[1], Q_cols[0])
        q11 = (Q_rows[1], Q_cols[1])
        
        r00 = (R_rows[0], R_cols[0])
        r01 = (R_rows[0], R_cols[1])
        r10 = (R_rows[1], R_cols[0])
        r11 = (R_rows[1], R_cols[1])
        
        # M1 = (P00+P11)*(Q00+Q11)
        u = {p00: 1, p11: 1}
        v = {q00: 1, q11: 1}
        w = {r00: 1, r01: 0, r10: 0, r11: 1}  # M1 contributes to R00 and R11
        # Actually, M1 contributes: R00 += M1, R11 += M1
        # But we need to be careful: each Strassen term is one rank-1 term
        # M1 appears in R00 and R11, so we need to handle this properly
        
        # Wait, I need to reconsider. In Strassen, each Mi is a product, and then
        # the result entries are linear combinations of the Mi's.
        # So each Mi is ONE rank-1 term in the tensor decomposition.
        # The W matrix for that term has the coefficients of how Mi contributes to each R entry.
        
        # Let me redo this properly.
        
        # M1 = (P00+P11)*(Q00+Q11)
        # Contribution to R: R00 += M1, R11 += M1
        u = {p00: 1, p11: 1}
        v = {q00: 1, q11: 1}
        w = {r00: 1, r11: 1}
        add_term(u, v, w)
        
        # M2 = (P10+P11)*Q00
        # Contribution: R10 += M2, R11 -= M2
        u = {p10: 1, p11: 1}
        v = {q00: 1}
        w = {r10: 1, r11: -1}
        add_term(u, v, w)
        
        # M3 = P00*(Q01-Q11)
        # Contribution: R01 += M3, R11 += M3
        u = {p00: 1}
        v = {q01: 1, q11: -1}
        w = {r01: 1, r11: 1}
        add_term(u, v, w)
        
        # M4 = P11*(Q10-Q00)
        # Contribution: R00 += M4, R10 += M4
        u = {p11: 1}
        v = {q10: 1, q00: -1}
        w = {r00: 1, r10: 1}
        add_term(u, v, w)
        
        # M5 = (P00+P01)*Q11
        # Contribution: R00 -= M5, R01 += M5
        u = {p00: 1, p01: 1}
        v = {q11: 1}
        w = {r00: -1, r01: 1}
        add_term(u, v, w)
        
        # M6 = (P10-P00)*(Q00+Q10)
        # Contribution: R11 += M6
        u = {p10: 1, p00: -1}
        v = {q00: 1, q10: 1}
        w = {r11: 1}
        add_term(u, v, w)
        
        # M7 = (P01-P11)*(Q01+Q11)
        # Contribution: R00 += M7
        u = {p01: 1, p11: -1}
        v = {q01: 1, q11: 1}
        w = {r00: 1}
        add_term(u, v, w)
    
    # Now apply to our 4 2x2*2x2 blocks
    
    # Block 1: A_top_left * B_tl -> contributes to C[0:2, 0:2]
    strassen_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: A_top_right * B_bl -> contributes to C[0:2, 0:2]
    strassen_2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Block 3: A_top_left * B_tr -> contributes to C[0:2, 2:4]
    strassen_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # Block 4: A_top_right * B_br -> contributes to C[0:2, 2:4]
    strassen_2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Part 2: Bottom 1 row of A (1x4) times B (4x4)
    # C[2,k] = sum_j A[2,j] * B[j,k]
    # This is 16 rank-1 terms
    for j in range(4):
        for k in range(4):
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(2, k): 1}
            add_term(u, v, w)
    
    # Now build U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

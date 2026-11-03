"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 104 decomposition
    # Split A (3x5) into A1 (3x2) and A2 (3x3)
    # Split B (5x7) into B1 (2x7) and B2 (3x7)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (3x2) * B1 (2x7)
    # Split B1 into B1a (2x2) and B1b (2x5)
    # A1 B1a: Rank 11 (Strassen for 2x2x2 + Schoolbook for 1x2x2)
    # A1 B1b: Rank 30 (Schoolbook 3x2x5)
    
    # Part 2: A2 (3x3) * B2 (3x7)
    # Split B2 into B2a (3x3) and B2b (3x4)
    # A2 B2a: Rank 27 (Schoolbook 3x3x3)
    # A2 B2b: Rank 36 (Schoolbook 3x3x4)
    
    # Total Rank: 11 + 30 + 27 + 36 = 104
    
    rank = 104
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(n_b, m_b, p_b, u_offset, v_offset, w_offset, gen_func):
        nonlocal col
        u_rows, v_rows, w_rows = n_b * m_b, m_b * p_b, p_b * n_b
        # gen_func returns (U, V, W) for the block with local indices
        # and the rank of the block
        U_b, V_b, W_b, r_b = gen_func()
        for i in range(u_rows):
            for c in range(r_b):
                U[u_offset + i][col + c] = U_b[i][c]
        for j in range(v_rows):
            for c in range(r_b):
                V[v_offset + j][col + c] = V_b[j][c]
        for k in range(w_rows):
            for c in range(r_b):
                W[w_offset + k][col + c] = W_b[k][c]
        col += r_b
        return r_b

    # --- Strassen 2x2x2 ---
    def strassen_222():
        r = 7
        U_b = [[0]*r for _ in range(4)]
        V_b = [[0]*r for _ in range(4)]
        W_b = [[0]*r for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U_b[0][0] = 1; U_b[3][0] = 1
        V_b[0][0] = 1; V_b[3][0] = 1
        W_b[0][0] = 1; W_b[1][0] = 1; W_b[2][0] = 1; W_b[3][0] = 1
        
        # M2 = (A21+A22)B11
        U_b[2][1] = 1; U_b[3][1] = 1
        V_b[0][1] = 1
        W_b[2][1] = 1; W_b[3][1] = 1
        
        # M3 = A11(B12-B22)
        U_b[0][2] = 1
        V_b[1][2] = 1; V_b[3][2] = -1
        W_b[1][2] = 1; W_b[3][2] = 1
        
        # M4 = A22(B21-B11)
        U_b[3][3] = 1
        V_b[2][3] = 1; V_b[0][3] = -1
        W_b[0][3] = 1; W_b[2][3] = 1
        
        # M5 = (A11+A12)B22
        U_b[0][4] = 1; U_b[1][4] = 1
        V_b[3][4] = 1
        W_b[0][4] = -1; W_b[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U_b[2][5] = 1; U_b[0][5] = -1
        V_b[0][5] = 1; V_b[1][5] = 1
        W_b[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U_b[1][6] = 1; U_b[3][6] = -1
        V_b[2][6] = 1; V_b[3][6] = 1
        W_b[0][6] = 1
        
        return U_b, V_b, W_b, r

    # --- Schoolbook n x m x p ---
    def schoolbook(n_b, m_b, p_b):
        r = n_b * m_b * p_b
        U_b = [[0]*r for _ in range(n_b * m_b)]
        V_b = [[0]*r for _ in range(m_b * p_b)]
        W_b = [[0]*r for _ in range(p_b * n_b)]
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i*m_b + j][c] = 1
                    V_b[j*p_b + k][c] = 1
                    W_b[k*n_b + i][c] = 1
                    c += 1
        return U_b, V_b, W_b, r

    # --- Composite 3x2x2 (Rank 11) ---
    def comp_322():
        # A (3x2), B (2x2)
        # Split A into A1 (2x2) and A2 (1x2)
        # C1 (2x2) = A1 B (Strassen)
        # C2 (1x2) = A2 B (Schoolbook 1x2x2)
        r_total = 11
        U_b = [[0]*r_total for _ in range(6)] # 3*2
        V_b = [[0]*r_total for _ in range(4)] # 2*2
        W_b = [[0]*r_total for _ in range(4)] # 2*3 (k*n+i, n=2? No, output is 2x3? No, C is 3x2? 
        # Wait, A is 3x2, B is 2x2. C is 3x2.
        # W indices: k (0..1), i (0..2). Index k*3+i.
        # My W_b size should be p*n = 2*3 = 6.
        # Let's re-verify dimensions.
        # Task: 3x5 by 5x7.
        # Part 1: A1 (3x2) by B1 (2x7). C1 (3x7).
        # Sub-part 1a: A1 (3x2) by B1a (2x2). C1a (3x2).
        # n=3, m=2, p=2.
        # U: 3*2=6 rows. V: 2*2=4 rows. W: 2*3=6 rows.
        
        U_b = [[0]*r_total for _ in range(6)]
        V_b = [[0]*r_total for _ in range(4)]
        W_b = [[0]*r_total for _ in range(6)]
        
        # Strassen part for A1 (rows 0,1)
        U_s, V_s, W_s, r_s = strassen_222()
        for i in range(4):
            for c in range(r_s):
                U_b[i][c] = U_s[i][c]
                V_b[i][c] = V_s[i][c]
                # W_s index: k*2+i (i in 0..1). 
                # W_b index: k*3+i (i in 0..1).
                # k=0: W_s[0] -> W_b[0], W_s[1] -> W_b[1]
                # k=1: W_s[2] -> W_b[3], W_s[3] -> W_b[4]
                # Wait, W_s is p*n = 2*2=4. Indices 0,1,2,3.
                # W_b is p*n = 2*3=6. Indices 0,1,2,3,4,5.
                # k=0, i=0 -> 0. k=0, i=1 -> 1.
                # k=1, i=0 -> 3. k=1, i=1 -> 4.
                # So W_s[0]->W_b[0], W_s[1]->W_b[1], W_s[2]->W_b[3], W_s[3]->W_b[4].
                for c in range(r_s):
                    if i == 0: W_b[0][c] = W_s[0][c]
                    if i == 1: W_b[1][c] = W_s[1][c]
                    if i == 2: W_b[3][c] = W_s[2][c]
                    if i == 3: W_b[4][c] = W_s[3][c]
        
        # Schoolbook part for A2 (row 2)
        U_sb, V_sb, W_sb, r_sb = schoolbook(1, 2, 2)
        # U_sb: 1*2=2 rows. V_sb: 2*2=4 rows. W_sb: 2*1=2 rows.
        # U_sb[0] corresponds to A[2,0], U_sb[1] to A[2,1].
        # In U_b, these are rows 4 and 5.
        # V_sb matches V_b (4 rows).
        # W_sb: k*1+i (i=0). k=0->0, k=1->1.
        # In W_b, i=2. k=0->2, k=1->5.
        for i in range(2):
            for c in range(r_sb):
                U_b[4+i][c + r_s] = U_sb[i][c]
                V_b[i][c + r_s] = V_sb[i][c]
                # W_sb index: k (0,1). W_b index: k*3+2.
                if i == 0: W_b[2][c + r_s] = W_sb[0][c]
                if i == 1: W_b[5][c + r_s] = W_sb[1][c]
                
        return U_b, V_b, W_b, r_total

    # 1. A1 (3x2) * B1a (2x2) -> Rank 11
    # U offset 0, V offset 0, W offset 0
    add_block(3, 2, 2, 0, 0, 0, comp_322)
    
    # 2. A1 (3x2) * B1b (2x5) -> Rank 30
    # U offset 0, V offset 2 (B1a was 2 cols, so next 5 cols start at 2), W offset 2 (C1a was 2 cols, so next 5 cols start at 2)
    # Wait, V is m*p. m=2. B1 is 2x7.
    # B1a is cols 0,1 of B1. B1b is cols 2,3,4,5,6 of B1.
    # In V (5x7), B1 occupies rows 0,1.
    # B1a: cols 0,1. B1b: cols 2..6.
    # So V offset for B1b is 2.
    # W is p*n. p=7, n=3.
    # C1a is cols 0,1 of C1. C1b is cols 2..6.
    # So W offset for C1b is 2.
    add_block(3, 2, 5, 0, 2, 2, lambda: schoolbook(3, 2, 5))
    
    # 3. A2 (3x3) * B2a (3x3) -> Rank 27
    # A2 is cols 2,3,4 of A (3x5).
    # B2 is rows 2,3,4 of B (5x7).
    # B2a is cols 0,1,2 of B2.
    # V offset: A2 cols 2..4, B2a cols 0..2.
    # V index: row j (2..4), col k (0..2).
    # Offset in V: row 2, col 0.
    # W offset: C2a cols 0..2.
    # Offset in W: col 0.
    add_block(3, 3, 3, 2, 0, 0, lambda: schoolbook(3, 3, 3))
    
    # 4. A2 (3x3) * B2b (3x4) -> Rank 36
    # B2b is cols 3..6 of B2.
    # V offset: row 2, col 3.
    # W offset: col 3.
    add_block(3, 3, 4, 2, 3, 3, lambda: schoolbook(3, 3, 4))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

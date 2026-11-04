import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=4 into 2+2.
    # Two sub-problems of 3x2x5.
    # Each 3x2x5 split i=1+2.
    #   1x2x5 (Rank 10) + 2x2x5 (Rank 18) = 28.
    # Total Rank 56.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term u, v, w
    def add_term(u, v, w):
        nonlocal col
        U[:, col] = u
        V[:, col] = v
        W[:, col] = w
        col += 1

    # Helper for Schoolbook 1x2x2
    # A: 1x2, B: 2x2, C: 1x2
    def schoolbook_1x2x2(base_u, base_v, base_w):
        # u: 1x2 indices in A
        # v: 2x2 indices in B
        # w: 1x2 indices in C
        # Terms: A_11 B_11, A_11 B_12, A_12 B_21, A_12 B_22
        # C_11 = A_11 B_11 + A_12 B_21
        # C_12 = A_11 B_12 + A_12 B_22
        
        # Term 1: A_11 B_11 -> C_11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1 # A_11
        v[base_v[0]] = 1 # B_11
        w[base_w[0]] = 1 # C_11
        add_term(u, v, w)
        
        # Term 2: A_12 B_21 -> C_11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[1]] = 1 # A_12
        v[base_v[2]] = 1 # B_21
        w[base_w[0]] = 1 # C_11
        add_term(u, v, w)

        # Term 3: A_11 B_12 -> C_12
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1 # A_11
        v[base_v[1]] = 1 # B_12
        w[base_w[1]] = 1 # C_12
        add_term(u, v, w)

        # Term 4: A_12 B_22 -> C_12
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[1]] = 1 # A_12
        v[base_v[3]] = 1 # B_22
        w[base_w[1]] = 1 # C_12
        add_term(u, v, w)

    # Helper for Schoolbook 1x2x1
    # A: 1x2, B: 2x1, C: 1x1
    def schoolbook_1x2x1(base_u, base_v, base_w):
        # A_11 B_11, A_12 B_21
        # C_11 = A_11 B_11 + A_12 B_21
        
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1
        v[base_v[0]] = 1
        w[base_w[0]] = 1
        add_term(u, v, w)

        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[1]] = 1
        v[base_v[1]] = 1
        w[base_w[0]] = 1
        add_term(u, v, w)

    # Helper for Strassen 2x2x2
    def strassen_2x2x2(base_u, base_v, base_w):
        # u: 2x2 indices (row-major)
        # v: 2x2 indices
        # w: 2x2 indices
        
        # P1 = (M11+M22)(N11+N22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1; u[base_u[3]] = 1
        v[base_v[0]] = 1; v[base_v[3]] = 1
        # C11 += 1, C22 += 1
        w[base_w[0]] = 1; w[base_w[3]] = 1
        add_term(u, v, w)
        
        # P2 = (M21+M22)N11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[2]] = 1; u[base_u[3]] = 1
        v[base_v[0]] = 1
        # C21 += 1, C22 += 1
        w[base_w[2]] = 1; w[base_w[3]] = 1
        add_term(u, v, w)

        # P3 = M11(N12-N22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1
        v[base_v[1]] = 1; v[base_v[3]] = -1
        # C12 += 1, C22 -= 1
        w[base_w[1]] = 1; w[base_w[3]] = -1
        add_term(u, v, w)

        # P4 = M22(N21-N11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[3]] = 1
        v[base_v[2]] = 1; v[base_v[0]] = -1
        # C11 -= 1, C21 += 1
        w[base_w[0]] = -1; w[base_w[2]] = 1
        add_term(u, v, w)

        # P5 = (M11+M12)N22
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1; u[base_u[1]] = 1
        v[base_v[3]] = 1
        # C12 += 1, C22 += 1
        w[base_w[1]] = 1; w[base_w[3]] = 1
        add_term(u, v, w)

        # P6 = (M21-M11)(N11+N12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[2]] = 1; u[base_u[0]] = -1
        v[base_v[0]] = 1; v[base_v[1]] = 1
        # C11 -= 1, C21 += 1
        w[base_w[0]] = -1; w[base_w[2]] = 1
        add_term(u, v, w)

        # P7 = (M12-M22)(N21+N22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[1]] = 1; u[base_u[3]] = -1
        v[base_v[2]] = 1; v[base_v[3]] = 1
        # C11 -= 1, C22 += 1
        w[base_w[0]] = -1; w[base_w[3]] = 1
        add_term(u, v, w)

    # Helper for Schoolbook 2x2x1
    # A: 2x2, B: 2x1, C: 2x1
    def schoolbook_2x2x1(base_u, base_v, base_w):
        # A_11 B_11, A_12 B_21, A_21 B_11, A_22 B_21
        # C_11 = A_11 B_11 + A_12 B_21
        # C_21 = A_21 B_11 + A_22 B_21
        
        # T1: A_11 B_11 -> C_11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[0]] = 1; v[base_v[0]] = 1; w[base_w[0]] = 1
        add_term(u, v, w)
        
        # T2: A_12 B_21 -> C_11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[1]] = 1; v[base_v[1]] = 1; w[base_w[0]] = 1
        add_term(u, v, w)
        
        # T3: A_21 B_11 -> C_21
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[2]] = 1; v[base_v[0]] = 1; w[base_w[1]] = 1
        add_term(u, v, w)
        
        # T4: A_22 B_21 -> C_21
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[base_u[3]] = 1; v[base_v[1]] = 1; w[base_w[1]] = 1
        add_term(u, v, w)

    # Process Block 1 (A1: rows 0-1 of A? No, A is 3x4. Split cols 0-1)
    # A1: 3x2 (cols 0,1). B1: 2x5 (rows 0,1).
    # C is 3x5.
    
    # Split i (rows of C, 3) into 1 + 2.
    # A1_row0: 1x2. A1_rows12: 2x2.
    
    # B1: 2x5. Split k (cols) into 2 + 2 + 1.
    # B1_01: 2x2. B1_23: 2x2. B1_4: 2x1.
    
    # Part 1: A1_row0 * B1 (1x2x5)
    # base_u for A1_row0: indices in A (3x4).
    # A1 is cols 0,1. Row 0.
    # Indices: A[0,0]=0, A[0,1]=1.
    base_u_1x2 = [0, 1]
    
    # B1_01: Rows 0,1. Cols 0,1.
    # B indices (2x5). Row 0: 0,1,2,3,4. Row 1: 5,6,7,8,9.
    # B1_01: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6.
    base_v_2x2_0 = [0, 1, 5, 6]
    
    # B1_23: Rows 0,1. Cols 2,3.
    # (0,2)->2, (0,3)->3, (1,2)->7, (1,3)->8.
    base_v_2x2_1 = [2, 3, 7, 8]
    
    # B1_4: Rows 0,1. Col 4.
    # (0,4)->4, (1,4)->9.
    base_v_2x1 = [4, 9]
    
    # C indices (3x5).
    # C1_row0: Row 0. Cols 0-4.
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[0,4]=4.
    
    # Sub 1.1: 1x2x2 (C cols 0,1)
    base_w_1x2_0 = [0, 1]
    schoolbook_1x2x2(base_u_1x2, base_v_2x2_0, base_w_1x2_0)
    
    # Sub 1.2: 1x2x2 (C cols 2,3)
    base_w_1x2_1 = [2, 3]
    schoolbook_1x2x2(base_u_1x2, base_v_2x2_1, base_w_1x2_1)
    
    # Sub 1.3: 1x2x1 (C col 4)
    base_w_1x1 = [4]
    schoolbook_1x2x1(base_u_1x2, base_v_2x1, base_w_1x1)
    
    # Part 2: A1_rows12 * B1 (2x2x5)
    # A1_rows12: Rows 1,2. Cols 0,1.
    # Indices: A[1,0]=4, A[1,1]=5, A[2,0]=8, A[2,1]=9.
    base_u_2x2 = [4, 5, 8, 9]
    
    # Sub 2.1: 2x2x2 (C cols 0,1)
    # C rows 1,2. Cols 0,1.
    # C[1,0]=5, C[1,1]=6, C[2,0]=10, C[2,1]=11.
    base_w_2x2_0 = [5, 6, 10, 11]
    strassen_2x2x2(base_u_2x2, base_v_2x2_0, base_w_2x2_0)
    
    # Sub 2.2: 2x2x2 (C cols 2,3)
    # C rows 1,2. Cols 2,3.
    # C[1,2]=7, C[1,3]=8, C[2,2]=12, C[2,3]=13.
    base_w_2x2_1 = [7, 8, 12, 13]
    strassen_2x2x2(base_u_2x2, base_v_2x2_1, base_w_2x2_1)
    
    # Sub 2.3: 2x2x1 (C col 4)
    # C rows 1,2. Col 4.
    # C[1,4]=9, C[2,4]=14.
    base_w_2x1 = [9, 14]
    schoolbook_2x2x1(base_u_2x2, base_v_2x1, base_w_2x1)
    
    # Process Block 2 (A2: cols 2,3. B2: rows 2,3)
    
    # Part 1: A2_row0 * B2 (1x2x5)
    # A2_row0: Row 0. Cols 2,3.
    # A[0,2]=2, A[0,3]=3.
    base_u_1x2 = [2, 3]
    
    # B2: Rows 2,3.
    # B indices: Row 2: 10,11,12,13,14. Row 3: 15,16,17,18,19.
    
    # B2_01: Cols 0,1.
    # (2,0)->10, (2,1)->11, (3,0)->15, (3,1)->16.
    base_v_2x2_0 = [10, 11, 15, 16]
    
    # B2_23: Cols 2,3.
    # (2,2)->12, (2,3)->13, (3,2)->17, (3,3)->18.
    base_v_2x2_1 = [12, 13, 17, 18]
    
    # B2_4: Col 4.
    # (2,4)->14, (3,4)->19.
    base_v_2x1 = [14, 19]
    
    # Sub 1.1: 1x2x2
    schoolbook_1x2x2(base_u_1x2, base_v_2x2_0, base_w_1x2_0)
    
    # Sub 1.2: 1x2x2
    schoolbook_1x2x2(base_u_1x2, base_v_2x2_1, base_w_1x2_1)
    
    # Sub 1.3: 1x2x1
    schoolbook_1x2x1(base_u_1x2, base_v_2x1, base_w_1x1)
    
    # Part 2: A2_rows12 * B2 (2x2x5)
    # A2_rows12: Rows 1,2. Cols 2,3.
    # A[1,2]=6, A[1,3]=7, A[2,2]=10, A[2,3]=11.
    base_u_2x2 = [6, 7, 10, 11]
    
    # Sub 2.1: 2x2x2
    strassen_2x2x2(base_u_2x2, base_v_2x2_0, base_w_2x2_0)
    
    # Sub 2.2: 2x2x2
    strassen_2x2x2(base_u_2x2, base_v_2x2_1, base_w_2x2_1)
    
    # Sub 2.3: 2x2x1
    schoolbook_2x2x1(base_u_2x2, base_v_2x1, base_w_2x1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

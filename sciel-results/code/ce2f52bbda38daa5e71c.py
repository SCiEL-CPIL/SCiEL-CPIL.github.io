import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition manually based on the block strategy.
    # Strategy:
    # Split m=5 into m1=2, m2=2, m3=1.
    # Split p=6 into p1=2, p2=2, p3=2 for the 2x2 blocks.
    
    # U shape: (n*m, rank) -> (10, rank)
    # V shape: (m*p, rank) -> (30, rank)
    # W shape: (p*n, rank) -> (12, rank)
    
    # We don't know the final rank until we build it, so we use lists.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx, v_idx, w_idx are lists of indices or single int?
        # Actually, U is a matrix. Each column corresponds to a rank term.
        # For a term, U has non-zeros at specific rows.
        # It's easier to build sparse columns.
        pass

    # Let's build the full matrices with zeros and fill them.
    # Max possible rank is 60. We expect 54.
    # To be safe, we can just append columns.
    
    # Helper to add a rank-1 term
    def add_rank1(U_rows, V_rows, W_rows):
        # U_rows: list of (row_index, coeff)
        # V_rows: list of (row_index, coeff)
        # W_rows: list of (row_index, coeff)
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for r, c in U_rows:
            u_col[r] += c
        for r, c in V_rows:
            v_col[r] += c
        for r, c in W_rows:
            w_col[r] += c
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Helper for Strassen M_2,2,2
    # Local A indices: 0,1,2,3 (row-major 2x2)
    # Local B indices: 0,1,2,3 (row-major 2x2)
    # Local C indices: 0,1,2,3 (row-major 2x2)
    # Global mapping:
    # A is 2x5. Block A_start_col is the offset in A's columns.
    # B is 5x6. Block B_start_row is the offset in B's rows, B_start_col is offset in B's cols.
    # C is 2x6. Block C_start_col is the offset in C's cols.
    
    def add_strassen(A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        # A is 2x2 block starting at col A_col_offset
        # B is 2x2 block starting at row B_row_offset, col B_col_offset
        # C is 2x2 block starting at col C_col_offset
        
        # A indices in global U (n*m = 10):
        # A[i, j] -> row i*m + j
        # i in 0..1, j in A_col_offset..A_col_offset+1
        
        # B indices in global V (m*p = 30):
        # B[i, j] -> row i*p + j
        # i in B_row_offset..B_row_offset+1, j in B_col_offset..B_col_offset+1
        
        # C indices in global W (p*n = 12):
        # C[i, j] -> row j*n + i (Task: W[k*n+i, r] where k is col, i is row)
        # So C[i, j] -> index j*n + i.
        # i in 0..1, j in C_col_offset..C_col_offset+1
        
        # Strassen terms:
        # m1 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # m2 = (c+d)e
        # c=A[1,0], d=A[1,1], e=B[0,0]
        # m3 = a(f-h)
        # a=A[0,0], f=B[0,1], h=B[1,1]
        # m4 = d(g-e)
        # d=A[1,1], g=B[1,0], e=B[0,0]
        # m5 = (a+b)h
        # a=A[0,0], b=A[0,1], h=B[1,1]
        # m6 = (c-a)(e+f)
        # c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
        # m7 = (b-d)(g+h)
        # b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
        
        # C entries:
        # p=C[0,0] = m1 + m4 - m5 + m7
        # q=C[0,1] = m3 + m5
        # r=C[1,0] = m2 + m4
        # s=C[1,1] = m1 - m2 + m3 + m6
        
        # We add 7 terms.
        
        # Term 1: m1
        U1 = [ (0*m + A_col_offset, 1), (1*m + A_col_offset+1, 1) ]
        V1 = [ (B_row_offset*p + B_col_offset, 1), ((B_row_offset+1)*p + B_col_offset+1, 1) ]
        W1 = [ (C_col_offset*n + 0, 1), ((C_col_offset+1)*n + 1, 1) ]
        add_rank1(U1, V1, W1)
        
        # Term 2: m2
        U2 = [ (1*m + A_col_offset, 1), (1*m + A_col_offset+1, 1) ]
        V2 = [ (B_row_offset*p + B_col_offset, 1) ]
        W2 = [ ((C_col_offset)*n + 1, 1) ] # r is C[1,0] -> index 0*n + 1? No.
        # C[i,j] -> j*n + i.
        # r is C[1,0]. i=1, j=0. Index 0*n + 1 = 1.
        # s is C[1,1]. i=1, j=1. Index 1*n + 1 = 3.
        # p is C[0,0]. i=0, j=0. Index 0.
        # q is C[0,1]. i=0, j=1. Index 1.
        # Wait, W1 had p (0) and s (3).
        # W2 should have r (1).
        # Let's re-verify W indices.
        # p=C[0,0] -> 0
        # q=C[0,1] -> 1
        # r=C[1,0] -> 1*2 + 0 = 2? No.
        # Task: W[k*n+i, r]. k is col index of C. i is row index of C.
        # C is n x p. Rows 0..n-1, Cols 0..p-1.
        # Index = k*n + i.
        # p = C[0,0]. k=0, i=0. Index 0.
        # q = C[0,1]. k=1, i=0. Index 1*2 + 0 = 2.
        # r = C[1,0]. k=0, i=1. Index 0*2 + 1 = 1.
        # s = C[1,1]. k=1, i=1. Index 1*2 + 1 = 3.
        
        # Correct W indices for 2x2 block starting at col C_col_offset:
        # p (C[0, C_col_offset]) -> (C_col_offset)*n + 0
        # q (C[0, C_col_offset+1]) -> (C_col_offset+1)*n + 0
        # r (C[1, C_col_offset]) -> (C_col_offset)*n + 1
        # s (C[1, C_col_offset+1]) -> (C_col_offset+1)*n + 1
        
        # Let's restart W assignments.
        # m1 contributes to p (+1) and s (+1)
        # m2 contributes to r (+1) and s (-1)
        # m3 contributes to q (+1) and s (+1)
        # m4 contributes to p (+1) and r (+1)
        # m5 contributes to p (-1) and q (+1)
        # m6 contributes to s (+1)
        # m7 contributes to p (+1)
        
        # I already added Term 1 (m1) with W1 = [p, s]. Correct.
        # Now Term 2 (m2):
        # U2: c, d -> A[1,0], A[1,1]
        # V2: e -> B[0,0]
        # W2: r (+1), s (-1)
        
        # I need to clear the previous add_rank1 call for m1? No, I can just build it correctly now.
        # I will rewrite the helper to take the full list of terms.
        
        pass

    # Let's implement the full logic cleanly.
    
    def add_strassen_block(A_col_off, B_row_off, B_col_off, C_col_off):
        # A indices:
        a = (0*m + A_col_off, 1)
        b = (0*m + A_col_off+1, 1)
        c = (1*m + A_col_off, 1)
        d = (1*m + A_col_off+1, 1)
        
        # B indices:
        e = (B_row_off*p + B_col_off, 1)
        f = (B_row_off*p + B_col_off+1, 1)
        g = ((B_row_off+1)*p + B_col_off, 1)
        h = ((B_row_off+1)*p + B_col_off+1, 1)
        
        # C indices:
        p_idx = (C_col_off*n + 0)
        q_idx = (C_col_off+1)*n + 0
        r_idx = (C_col_off*n + 1)
        s_idx = (C_col_off+1)*n + 1
        
        # Terms:
        # 1. m1 = (a+d)(e+h) -> p+1, s+1
        add_rank1([a, d], [e, h], [(p_idx, 1), (s_idx, 1)])
        
        # 2. m2 = (c+d)e -> r+1, s-1
        add_rank1([c, d], [e], [(r_idx, 1), (s_idx, -1)])
        
        # 3. m3 = a(f-h) -> q+1, s+1
        add_rank1([a], [f, (B_row_off*p + B_col_off+1, -1)], [(q_idx, 1), (s_idx, 1)])
        
        # 4. m4 = d(g-e) -> p+1, r+1
        add_rank1([d], [g, (B_row_off*p + B_col_off, -1)], [(p_idx, 1), (r_idx, 1)])
        
        # 5. m5 = (a+b)h -> p-1, q+1
        add_rank1([a, b], [h], [(p_idx, -1), (q_idx, 1)])
        
        # 6. m6 = (c-a)(e+f) -> s+1
        add_rank1([(1*m + A_col_off, 1), (0*m + A_col_off, -1)], [e, f], [(s_idx, 1)])
        
        # 7. m7 = (b-d)(g+h) -> p+1
        add_rank1([(0*m + A_col_off+1, 1), (1*m + A_col_off+1, -1)], [g, h], [(p_idx, 1)])

    # Add M_2,2,6 blocks
    # Block 1: A cols 0,1. B rows 0,1.
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 2)
    add_strassen_block(0, 0, 4, 4)
    
    # Block 2: A cols 2,3. B rows 2,3.
    add_strassen_block(2, 2, 0, 0)
    add_strassen_block(2, 2, 2, 2)
    add_strassen_block(2, 2, 4, 4)
    
    # Block 3: A col 4. B row 4.
    # M_2,1,6
    # A[0,4], A[1,4]
    # B[4,0]..B[4,5]
    # C[0,0..5], C[1,0..5]
    for j in range(6):
        # Term: A[i,4] * B[4,j] -> C[i,j]
        # We can combine i=0 and i=1 into separate terms?
        # Rank 12.
        # Term for i=0: A[0,4] * B[4,j] -> C[0,j]
        add_rank1([(0*m + 4, 1)], [(4*p + j, 1)], [((j*n + 0), 1)])
        # Term for i=1: A[1,4] * B[4,j] -> C[1,j]
        add_rank1([(1*m + 4, 1)], [(4*p + j, 1)], [((j*n + 1), 1)])

    # Convert lists to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # The task requires integer/half-integer entries. Our entries are 1, -1, 0.
    # Return as lists of lists to match the seed format?
    # Seed returns {"U": U, "V": V, "W": W} where U is list of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

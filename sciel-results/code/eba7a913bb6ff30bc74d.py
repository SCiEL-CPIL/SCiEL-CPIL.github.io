"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition strategy.
    # A is 3x4, B is 4x6.
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2).
    # C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    # A1*Bk is 2x2x2 (Strassen, rank 7)
    # A2*Bk is 1x2x2 (Rank 2)
    
    # Total rank = 3 * 7 + 3 * 2 = 21 + 6 = 27? 
    # Wait, A1 is 2x4, Bk is 4x2.
    # A1*Bk is 2x4 * 4x2 = 2x2.
    # This is a 2x4x2 multiplication.
    # Schoolbook rank 16.
    # Can we do better?
    # Split A1 into A11 (2x2), A12 (2x2).
    # Split Bk into Bk1 (2x2), Bk2 (2x2).
    # A1*Bk = A11*Bk1 + A12*Bk2.
    # Each is 2x2x2. Rank 7.
    # So A1*Bk rank 14.
    # 3 such blocks: 3 * 14 = 42.
    
    # A2 is 1x4. Bk is 4x2.
    # A2*Bk is 1x4 * 4x2 = 1x2.
    # Split A2 into A21 (1x2), A22 (1x2).
    # Split Bk into Bk1 (2x2), Bk2 (2x2).
    # A2*Bk = A21*Bk1 + A22*Bk2.
    # Each is 1x2x2. Rank 2.
    # So A2*Bk rank 4.
    # 3 such blocks: 3 * 4 = 12.
    
    # Total rank = 42 + 12 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i_offset, j_offset, k_offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols are indices in A
        # b_rows, b_cols are indices in B
        # c_rows, c_cols are indices in C
        # We assume the block is 2x2 for A, 2x2 for B, 2x2 for C
        
        # Strassen mults
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map the linear combinations to U, V, W vectors.
        # U vector has non-zeros at A indices.
        # V vector has non-zeros at B indices.
        # W vector has non-zeros at C indices.
        
        # For each mult m, we define U_m, V_m, W_m.
        # U_m is a vector of length n*m.
        # V_m is a vector of length m*p.
        # W_m is a vector of length p*n.
        
        # M1
        U_m = [0] * (n * m)
        V_m = [0] * (m * p)
        W_m = [0] * (p * n)
        
        # A11 at (i_offset, j_offset), A22 at (i_offset+1, j_offset+1)
        # B11 at (j_offset, k_offset), B22 at (j_offset+1, k_offset+1)
        # C11 at (k_offset, i_offset), C12 at (k_offset, i_offset+1)
        # C21 at (k_offset+1, i_offset), C22 at (k_offset+1, i_offset+1)
        
        # Note: A index is i*m + j. B index is j*p + k. C index is k*n + i.
        
        def set_U(idx, val):
            U_m[idx] += val
        def set_V(idx, val):
            V_m[idx] += val
        def set_W(idx, val):
            W_m[idx] += val
            
        # M1
        set_U(i_offset * m + j_offset, 1)
        set_U((i_offset + 1) * m + (j_offset + 1), 1)
        set_V(j_offset * p + k_offset, 1)
        set_V((j_offset + 1) * p + (k_offset + 1), 1)
        # C contributions: C11 (+1), C22 (+1)
        set_W(k_offset * n + i_offset, 1)
        set_W((k_offset + 1) * n + (i_offset + 1), 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M2: (A21 + A22) B11
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U((i_offset + 1) * m + j_offset, 1)
        set_U((i_offset + 1) * m + (j_offset + 1), 1)
        set_V(j_offset * p + k_offset, 1)
        # C21 (+1)
        set_W((k_offset + 1) * n + i_offset, 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M3: A11 (B12 - B22)
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U(i_offset * m + j_offset, 1)
        set_V(j_offset * p + (k_offset + 1), 1)
        set_V((j_offset + 1) * p + (k_offset + 1), -1)
        # C12 (+1)
        set_W(k_offset * n + (i_offset + 1), 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M4: A22 (B21 - B11)
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U((i_offset + 1) * m + (j_offset + 1), 1)
        set_V((j_offset + 1) * p + k_offset, 1)
        set_V(j_offset * p + k_offset, -1)
        # C11 (+1), C21 (+1)
        set_W(k_offset * n + i_offset, 1)
        set_W((k_offset + 1) * n + i_offset, 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M5: (A11 + A12) B22
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U(i_offset * m + j_offset, 1)
        set_U(i_offset * m + (j_offset + 1), 1)
        set_V((j_offset + 1) * p + (k_offset + 1), 1)
        # C11 (-1), C12 (+1)
        set_W(k_offset * n + i_offset, -1)
        set_W(k_offset * n + (i_offset + 1), 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M6: (A21 - A11)(B11 + B12)
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U((i_offset + 1) * m + j_offset, 1)
        set_U(i_offset * m + j_offset, -1)
        set_V(j_offset * p + k_offset, 1)
        set_V(j_offset * p + (k_offset + 1), 1)
        # C22 (+1)
        set_W((k_offset + 1) * n + (i_offset + 1), 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M7: (A12 - A22)(B21 + B22)
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        set_U(i_offset * m + (j_offset + 1), 1)
        set_U((i_offset + 1) * m + (j_offset + 1), -1)
        set_V((j_offset + 1) * p + k_offset, 1)
        set_V((j_offset + 1) * p + (k_offset + 1), 1)
        # C11 (+1)
        set_W(k_offset * n + i_offset, 1)
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1

    # Helper to add a 1x2x2 block (Rank 2)
    # A is 1x2, B is 2x2, C is 1x2
    def add_1x2x2(i_offset, j_offset, k_offset):
        nonlocal col
        # A1 at (i_offset, j_offset), A2 at (i_offset, j_offset+1)
        # B11 at (j_offset, k_offset), B12 at (j_offset, k_offset+1)
        # B21 at (j_offset+1, k_offset), B22 at (j_offset+1, k_offset+1)
        # C1 at (k_offset, i_offset), C2 at (k_offset+1, i_offset)
        
        # C1 = A1*B11 + A2*B21
        # C2 = A1*B12 + A2*B22
        
        # We use 2 mults?
        # M1 = A1 * (B11 + B12)? No.
        # Rank 2 decomposition:
        # M1 = (A1 + A2) * (B11 + B21)?
        # Let's just use the 4 scalar mults if rank 2 is hard to construct simply?
        # No, rank 2 is known.
        # But implementing the specific linear forms for rank 2 1x2x2 is tricky.
        # Let's use schoolbook for 1x2x2 (rank 4) to be safe and simple?
        # 3 * 4 = 12.
        # Total 42 + 12 = 54.
        
        # M1: A1 B11
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        U_m[i_offset * m + j_offset] = 1
        V_m[j_offset * p + k_offset] = 1
        W_m[k_offset * n + i_offset] = 1
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M2: A1 B12
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        U_m[i_offset * m + j_offset] = 1
        V_m[j_offset * p + (k_offset + 1)] = 1
        W_m[(k_offset + 1) * n + i_offset] = 1
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M3: A2 B21
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        U_m[i_offset * m + (j_offset + 1)] = 1
        V_m[(j_offset + 1) * p + k_offset] = 1
        W_m[k_offset * n + i_offset] = 1
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1
        
        # M4: A2 B22
        U_m = [0] * (n * m); V_m = [0] * (m * p); W_m = [0] * (p * n)
        U_m[i_offset * m + (j_offset + 1)] = 1
        V_m[(j_offset + 1) * p + (k_offset + 1)] = 1
        W_m[(k_offset + 1) * n + i_offset] = 1
        U[col] = U_m; V[col] = V_m; W[col] = W_m
        col += 1

    # Apply blocks
    # A1 is rows 0-1, cols 0-3
    # A2 is row 2, cols 0-3
    # B is rows 0-3, cols 0-5
    
    # B1: cols 0-1
    # B2: cols 2-3
    # B3: cols 4-5
    
    # A1*B1: i=0, j=0, k=0
    add_strassen(0, 0, 0)
    # A1*B2: i=0, j=0, k=2
    add_strassen(0, 0, 2)
    # A1*B3: i=0, j=0, k=4
    add_strassen(0, 0, 4)
    
    # A2*B1: i=2, j=0, k=0
    add_1x2x2(2, 0, 0)
    # A2*B2: i=2, j=0, k=2
    add_1x2x2(2, 0, 2)
    # A2*B3: i=2, j=0, k=4
    add_1x2x2(2, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Dimensions
    n, m, p = 2, 3, 7
    rank = 39
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a simple tensor term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    # Helper for Strassen 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        # We can't just do one column for sums of inputs in U/V.
        # The decomposition must be linear in A and B.
        # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # This requires 4 terms?
        # No, Strassen works by defining new matrices.
        # But the rank counts the number of multiplications of scalars (or blocks).
        # Here, we are decomposing the tensor.
        # The tensor for 2x2x2 multiplication is a sum of 7 simple tensors?
        # No, the rank is 7.
        # The 7 terms are linear combinations of the input elements.
        
        # U columns will have coefficients for A elements.
        # V columns will have coefficients for B elements.
        # W columns will have coefficients for C elements.
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=-1, C10=-1, C11=1  (Wait, C00 = M1 + ... so M1 contributes +1 to C00? No.)
        
        # Let's trace the contributions to C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns.
        
        # Col 1: M1 = (A00+A11)(B00+B11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c01][col] = -1; W[c10][col] = -1; W[c11][col] = 1
        col += 1
        
        # Col 2: M2 = (A10+A11) B00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # Col 3: M3 = A00 (B01-B11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # Col 4: M4 = A11 (B10-B00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # Col 5: M5 = (A00+A01) B11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # Col 6: M6 = (A10-A00) (B00+B01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Col 7: M7 = (A01-A11) (B10+B11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = -1
        col += 1

    # Helper for 2x2x1 (Rank 4)
    def mult_221(a00, a01, a10, a11, b00, b10, c00, c10):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00
        U[a00][col] = 1
        V[b00][col] = 1
        W[c00][col] = 1
        col += 1
        
        # Term 2: A01 B10
        U[a01][col] = 1
        V[b10][col] = 1
        W[c00][col] = 1
        col += 1
        
        # Term 3: A10 B00
        U[a10][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # Term 4: A11 B10
        U[a11][col] = 1
        V[b10][col] = 1
        W[c10][col] = 1
        col += 1

    # Helper for 2x1xp (Rank 2p)
    # A is 2x1, B is 1xp.
    def mult_21p(a0, a1, b_indices, c_indices):
        nonlocal col
        # C_i,k = A_i B_k
        for k in range(len(b_indices)):
            b_idx = b_indices[k]
            c0_idx = c_indices[k] # i=0
            c1_idx = c_indices[k] + 1 # i=1? 
            # Wait, C index is k*n + i.
            # If c_indices are the base indices for i=0, then i=1 is +1.
            
            # Term A0 * Bk
            U[a0][col] = 1
            V[b_idx][col] = 1
            W[c0_idx][col] = 1
            col += 1
            
            # Term A1 * Bk
            U[a1][col] = 1
            V[b_idx][col] = 1
            W[c1_idx][col] = 1
            col += 1

    # --- Part 2: A_R B_B (2x1x7) ---
    # A_R: rows 0,1 col 2. Indices: 0*3+2=2, 1*3+2=5.
    # B_B: row 2 cols 0..6. Indices: 2*7+k = 14+k.
    # C_2: k 0..6, i 0..1. Indices: k*2 + i.
    a_r0 = 2
    a_r1 = 5
    b_b_indices = [14 + k for k in range(7)]
    c_2_indices = [2 * k + 0 for k in range(7)] # i=0 base
    # mult_21p expects a0, a1, list of b, list of c (i=0 bases)
    mult_21p(a_r0, a_r1, b_b_indices, c_2_indices)
    
    # --- Part 1: A_L B_T (2x2x7) ---
    # A_L: rows 0,1 cols 0,1.
    # a00=0, a01=1, a10=3, a11=4.
    
    # B_T: rows 0,1 cols 0..6.
    # B_T,L: cols 0,1.
    # B_T,R: cols 2..6.
    
    # --- Part 1.1: A_L B_T,L (2x2x2) ---
    # B_T,L indices:
    # b00 = 0*7+0 = 0
    # b01 = 0*7+1 = 1
    # b10 = 1*7+0 = 7
    # b11 = 1*7+1 = 8
    # C_1,1 indices (k 0..1, i 0..1):
    # c00 = 0*2+0 = 0
    # c01 = 1*2+0 = 2  (k=1, i=0)
    # c10 = 0*2+1 = 1  (k=0, i=1)
    # c11 = 1*2+1 = 3  (k=1, i=1)
    strassen_222(0, 1, 3, 4, 0, 1, 7, 8, 0, 2, 1, 3)
    
    # --- Part 1.2: A_L B_T,R (2x2x5) ---
    # B_T,R: cols 2..6.
    # Split into B_TR,L (cols 2,3) and B_TR,R (cols 4,5,6).
    
    # --- Part 1.2.1: A_L B_TR,L (2x2x2) ---
    # B_TR,L indices:
    # b00 = 0*7+2 = 2
    # b01 = 0*7+3 = 3
    # b10 = 1*7+2 = 9
    # b11 = 1*7+3 = 10
    # C_1,2,1 indices (k 2..3, i 0..1):
    # k=2: c00=4, c10=5
    # k=3: c01=6, c11=7
    # Note: c01 here corresponds to C[0,3], which is index 3*2+0=6.
    # c11 here corresponds to C[1,3], which is index 3*2+1=7.
    # My strassen_222 function takes c00, c01, c10, c11 corresponding to C[0,0], C[0,1], C[1,0], C[1,1].
    # So c00=4, c01=6, c10=5, c11=7.
    strassen_222(0, 1, 3, 4, 2, 3, 9, 10, 4, 6, 5, 7)
    
    # --- Part 1.2.2: A_L B_TR,R (2x2x3) ---
    # B_TR,R: cols 4,5,6.
    # Split into B_TRR,L (cols 4,5) and B_TRR,R (col 6).
    
    # --- Part 1.2.2.1: A_L B_TRR,L (2x2x2) ---
    # B_TRR,L indices:
    # b00 = 0*7+4 = 4
    # b01 = 0*7+5 = 5
    # b10 = 1*7+4 = 11
    # b11 = 1*7+5 = 12
    # C_1,2,2,1 indices (k 4..5, i 0..1):
    # k=4: c00=8, c10=9
    # k=5: c01=10, c11=11
    strassen_222(0, 1, 3, 4, 4, 5, 11, 12, 8, 10, 9, 11)
    
    # --- Part 1.2.2.2: A_L B_TRR,R (2x2x1) ---
    # B_TRR,R: col 6.
    # b00 = 0*7+6 = 6
    # b10 = 1*7+6 = 13
    # C_1,2,2,2 indices (k 6, i 0..1):
    # k=6: c00=12, c10=13
    mult_221(0, 1, 3, 4, 6, 13, 12, 13)

    return {"U": U, "V": V, "W": W}

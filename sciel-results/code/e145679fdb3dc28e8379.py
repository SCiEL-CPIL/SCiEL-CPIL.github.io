"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-44 decomposition for 3x4x4 matrix multiplication.
    # Top 2x4x4 part: 4 independent 2x2x2 multiplications via Strassen (4*7=28 terms)
    # Bottom 1x4x4 part: schoolbook (16 terms)
    # Total rank: 44
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    
    # U: (n*m, rank) = (12, 44)
    # V: (m*p, rank) = (16, 44)
    # W: (p*n, rank) = (12, 44)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B10-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The 4 products correspond to:
    # Product 0: rows 0-1 of A (top), cols 0-1 of B (top-left) -> C[0:2, 0:2]
    # Product 1: rows 0-1 of A (top), cols 2-3 of B (top-right) -> C[0:2, 2:4]
    # Product 2: rows 0-1 of A (top), cols 0-1 of B (bottom-left) -> C[2:4, 0:2]
    # Product 3: rows 0-1 of A (top), cols 2-3 of B (bottom-right) -> C[2:4, 2:4]
    
    # For product t (0..3):
    # A row offset: 0 (both products use top 2 rows of A)
    # B col offset: t*2 (product 0: cols 0-1, product 1: cols 2-3, etc.)
    # Wait, B is 4x4, so cols 0-3. Product 0 uses B cols 0-1, product 1 uses B cols 2-3.
    # But the output C is 3x4. Rows 0-1 of C come from top 2 rows of A.
    # C[0:2, 0:2] = A[0:2, :4] * B[:4, 0:2]  -- no, that's not right.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # For i in {0,1}, j in {0,1}: C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # This splits into sum_{k=0}^{1} + sum_{k=2}^{3}
    # = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # So it's 2 products of 2x2x2, not 4.
    
    # Let me reconsider. The 3x4x4 multiplication can be decomposed as:
    # C[0:2, :] = A[0:2, :] * B[:, :]  (2x4x4)
    # C[2:4, :] = A[2:4, :] * B[:, :]  (1x4x4)
    
    # For 2x4x4: we can split B into left and right halves:
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    
    # So we need 4 products of 2x2x2 for the top part.
    # Product 0: A[0:2,0:2] * B[0:2,0:2] -> contributes to C[0:2,0:2]
    # Product 1: A[0:2,2:4] * B[2:4,0:2] -> contributes to C[0:2,0:2]
    # Product 2: A[0:2,0:2] * B[0:2,2:4] -> contributes to C[0:2,2:4]
    # Product 3: A[0:2,2:4] * B[2:4,2:4] -> contributes to C[0:2,2:4]
    
    # For each product t, we have 2x2 matrices:
    # A_t is 2x2, B_t is 2x2, result is 2x2
    
    # Product 0: A0 = A[0:2,0:2], B0 = B[0:2,0:2], output to C[0:2,0:2]
    # Product 1: A1 = A[0:2,2:4], B1 = B[2:4,0:2], output to C[0:2,0:2]
    # Product 2: A2 = A[0:2,0:2], B2 = B[0:2,2:4], output to C[0:2,2:4]
    # Product 3: A3 = A[0:2,2:4], B3 = B[2:4,2:4], output to C[0:2,2:4]
    
    # For product t with 2x2 A_t and B_t:
    # A_t indices: row r (0,1), col c (0,1) -> global A row r, col depends on t
    # B_t indices: row r (0,1), col c (0,1) -> global B row depends on t, col depends on t
    
    # Let's define for each product t:
    # a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset
    
    # Product 0: a_row=0, a_col=0, b_row=0, b_col=0, c_row=0, c_col=0
    # Product 1: a_row=0, a_col=2, b_row=2, b_col=0, c_row=0, c_col=0
    # Product 2: a_row=0, a_col=0, b_row=0, b_col=2, c_row=0, c_col=2
    # Product 3: a_row=0, a_col=2, b_row=2, b_col=2, c_row=0, c_col=2
    
    products = [
        # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2),
    ]
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B10-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Strassen product s (0..6), and for each matrix product t (0..3):
    # We need to set up U, V, W entries.
    
    # Strassen product definitions (local 2x2 indices):
    # s=0 (M1): A entries: (0,0):+1, (1,1):+1; B entries: (0,0):+1, (1,1):+1; C: all +1
    # s=1 (M2): A entries: (1,0):+1, (1,1):+1; B entries: (0,0):+1; C: (0,0):+1, (0,1):+1, (1,0):+1, (1,1):+1
    # Wait, M2 = (A10+A11)B00. This gives a 1x2 vector times 2x1? No.
    # M2 is a scalar: (A10+A11) * B00. So it's rank 1.
    # In the Strassen algorithm, each M_i is a scalar (rank 1).
    # M1 = (A00+A11)(B00+B11) -> scalar
    # M2 = (A10+A11)B00 -> scalar
    # M3 = A00(B11-B00) -> scalar
    # M4 = A11(B10-B01) -> scalar
    # M5 = (A00+A01)B11 -> scalar
    # M6 = (A10-A00)(B00+B01) -> scalar
    # M7 = (A01-A11)(B10+B11) -> scalar
    
    # For each scalar M_s, we have:
    # U entries: for each (r,c) in A_t that contributes, add coefficient to U[global_a_idx, col]
    # V entries: for each (r,c) in B_t that contributes, add coefficient to V[global_b_idx, col]
    # W entries: for each (r,c) in C_t that contributes, add coefficient to W[global_c_idx, col]
    
    # Strassen scalar definitions:
    # s=0: A: {(0,0):+1, (1,1):+1}, B: {(0,0):+1, (1,1):+1}, C: {(0,0):+1, (0,1):+1, (1,0):+1, (1,1):+1}
    # s=1: A: {(1,0):+1, (1,1):+1}, B: {(0,0):+1}, C: {(0,0):+1, (0,1):+1, (1,0):+1, (1,1):+1}
    # Wait, M2 = (A10+A11)*B00. The result is a scalar, but in Strassen, this scalar is used to compute multiple C entries.
    # Actually in Strassen, M2 contributes to C10 and C11.
    # C10 = M2 + M3 + M4 - M6
    # C11 = M2 + M5 + M4 - M6 - M7
    # C00 = M1 - M3 + M6 + M7
    # C01 = M1 - M2 + M5 + M7
    
    # So for each M_s, we need to know which C entries it contributes to and with what sign.
    
    # Strassen reconstruction:
    # C00 = M1 - M3 + M6 + M7
    # C01 = M1 - M2 + M5 + M7
    # C10 = M2 + M3 + M4 - M6
    # C11 = M2 + M5 + M4 - M6 - M7
    
    # So W contributions:
    # s=0 (M1): C00:+1, C01:+1
    # s=1 (M2): C01:-1, C10:+1, C11:+1
    # s=2 (M3): C00:-1, C10:+1
    # s=3 (M4): C10:+1, C11:+1
    # s=4 (M5): C01:+1, C11:+1
    # s=5 (M6): C00:+1, C10:-1, C11:-1
    # s=6 (M7): C00:+1, C01:+1, C11:-1
    
    strassen_U = [
        # s=0: A00+A11
        {(0,0): 1, (1,1): 1},
        # s=1: A10+A11
        {(1,0): 1, (1,1): 1},
        # s=2: A00
        {(0,0): 1},
        # s=3: A11
        {(1,1): 1},
        # s=4: A00+A01
        {(0,0): 1, (0,1): 1},
        # s=5: A10-A00
        {(1,0): 1, (0,0): -1},
        # s=6: A01-A11
        {(0,1): 1, (1,1): -1},
    ]
    
    strassen_V = [
        # s=0: B00+B11
        {(0,0): 1, (1,1): 1},
        # s=1: B00
        {(0,0): 1},
        # s=2: B11-B00
        {(1,1): 1, (0,0): -1},
        # s=3: B10-B01
        {(1,0): 1, (0,1): -1},
        # s=4: B11
        {(1,1): 1},
        # s=5: B00+B01
        {(0,0): 1, (0,1): 1},
        # s=6: B10+B11
        {(1,0): 1, (1,1): 1},
    ]
    
    strassen_W = [
        # s=0: C00:+1, C01:+1
        {(0,0): 1, (0,1): 1},
        # s=1: C01:-1, C10:+1, C11:+1
        {(0,1): -1, (1,0): 1, (1,1): 1},
        # s=2: C00:-1, C10:+1
        {(0,0): -1, (1,0): 1},
        # s=3: C10:+1, C11:+1
        {(1,0): 1, (1,1): 1},
        # s=4: C01:+1, C11:+1
        {(0,1): 1, (1,1): 1},
        # s=5: C00:+1, C10:-1, C11:-1
        {(0,0): 1, (1,0): -1, (1,1): -1},
        # s=6: C00:+1, C01:+1, C11:-1
        {(0,0): 1, (0,1): 1, (1,1): -1},
    ]
    
    col = 0
    # Top 2x4x4 part: 4 products * 7 Strassen terms = 28 columns
    for t in range(4):
        a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off = products[t]
        for s in range(7):
            # U entries
            for (lr, lc), coeff in strassen_U[s].items():
                # Local A row lr (0,1), col lc (0,1)
                # Global A row: a_row_off + lr, col: a_col_off + lc
                # U index: row * m + col = (a_row_off + lr) * m + (a_col_off + lc)
                u_idx = (a_row_off + lr) * m + (a_col_off + lc)
                U[u_idx][col] += coeff
            
            # V entries
            for (lr, lc), coeff in strassen_V[s].items():
                # Local B row lr (0,1), col lc (0,1)
                # Global B row: b_row_off + lr, col: b_col_off + lc
                # V index: row * p + col = (b_row_off + lr) * p + (b_col_off + lc)
                v_idx = (b_row_off + lr) * p + (b_col_off + lc)
                V[v_idx][col] += coeff
            
            # W entries
            for (lr, lc), coeff in strassen_W[s].items():
                # Local C row lr (0,1), col lc (0,1)
                # Global C row: c_row_off + lr, col: c_col_off + lc
                # W index: row * n + col = (c_col_off + lc) * n + (c_row_off + lr)
                # W[k*n+i, r] where k is col of C, i is row of C
                w_idx = (c_col_off + lc) * n + (c_row_off + lr)
                W[w_idx][col] += coeff
            
            col += 1
    
    # Bottom 1x4x4 part: schoolbook, 16 terms
    # C[2, j] = sum_{k=0}^{3} A[2, k] * B[k, j] for j in 0..3
    for k in range(4):
        for j in range(4):
            # U: A[2, k] -> index 2*m + k = 8 + k
            U[2 * m + k][col] = 1
            # V: B[k, j] -> index k*p + j
            V[k * p + j][col] = 1
            # W: C[2, j] -> index j*n + 2
            W[j * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

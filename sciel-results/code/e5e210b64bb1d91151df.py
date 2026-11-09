import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, j, k, w_val):
        nonlocal col
        if col >= rank:
            return
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = w_val
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = w_val
        col += 1

    def add_strassen_block(i0, j0, k0):
        # i0, j0, k0 are starting indices for 2x2x2 block
        # A rows i0, i0+1; Cols j0, j0+1
        # B Rows j0, j0+1; Cols k0, k0+1
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U terms for A: (1,0), (1,1) relative to i0,j0
        # V terms for B: (0,0), (1,1) relative to j0,k0
        
        # We need to add to U, V, W for each of the 7 products.
        # The products are linear combinations.
        # Instead of pre-computing linear combos, we can add terms with coefficients.
        # But the interface requires U,V,W entries.
        # The sum is sum_c U_ic V_ck W_ki.
        # If we add a term for M1, it is U1 * V1 * W1.
        # U1 must represent (A00+A11). So U1 has 1 at i0,j0 and 1 at i0+1, j0+1.
        # But U is a matrix. We can put 1s in multiple rows of the same column.
        # Yes, U is (nm, rank).
        
        # Term 1: (A00+A11)(B00+B11)
        # U col: 1 at i0*m+j0, 1 at (i0+1)*m+(j0+1)
        # V col: 1 at j0*p+k0, 1 at (j0+1)*p+(k0+1)
        # W col: 1 at k0*n+i0, 1 at k1*n+i0? No.
        # The result C is 2x2.
        # C00 gets -M1?
        # C11 gets +M1?
        # We need to distribute the coefficient to the W terms.
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5 - M6
        # C10 = M2 + M4 - M7
        # C11 = M1 - M3 + M2 - M5
        
        # We add 7 columns to U, V, W.
        # For each column t (0..6), we define U_t, V_t, W_t.
        
        # t=0: M1
        # U: A00 + A11
        # V: B00 + B11
        # W: Coeffs for C00 (+1), C11 (+1)
        
        # This logic is correct.
        
        def add_block_term(u_indices, v_indices, w_indices, w_vals):
            nonlocal col
            if col >= rank:
                return
            for idx, val in zip(u_indices, [1]*len(u_indices)):
                U[idx][col] += val
            for idx, val in zip(v_indices, [1]*len(v_indices)):
                V[idx][col] += val
            for idx, val in zip(w_indices, w_vals):
                W[idx][col] += val
            col += 1

        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1 = (A00+A11)(B00+B11)
        add_block_term([u00, u11], [v00, v11], [w00, w11], [1, 1])
        
        # M2 = (A10+A11)B00
        add_block_term([u10, u11], [v00], [w10, w11], [1, 1]) # C10 gets M2, C11 gets M2?
        # Check formula: C10 = M2 + M4 - M7. C11 = M1 - M3 + M2 - M5.
        # So M2 contributes to C10 (+1) and C11 (+1).
        
        # M3 = A00(B01-B11)
        add_block_term([u00], [v01, v11], [w01, w11], [1, -1]) # C01 gets M3, C11 gets -M3
        # C01 = M3 + M5 - M6. C11 = M1 - M3 + M2 - M5.
        
        # M4 = A11(B10-B00)
        add_block_term([u11], [v10, v00], [w10, w00], [1, -1]) # C10 gets M4, C00 gets -M4
        # C10 = M2 + M4 - M7. C00 = M1 - M4 + M6 - M7.
        
        # M5 = (A00+A01)B11
        add_block_term([u00, u01], [v11], [w01, w11], [1, -1]) # C01 gets M5, C11 gets -M5
        # C01 = M3 + M5 - M6. C11 = M1 - M3 + M2 - M5.
        
        # M6 = (A10-A00)(B00+B01)
        add_block_term([u10, u00], [v00, v01], [w10, w01], [1, -1]) # C10 gets M6, C01 gets -M6
        # C10 = M2 + M4 - M7. Wait, M6 in C10?
        # Standard: C10 = M2 + M4 - M7. No M6.
        # C01 = M3 + M5 - M6.
        # Let me re-verify Strassen reconstruction.
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5 - M6
        # C10 = M2 + M4 - M7
        # C11 = M1 - M3 + M2 - M5
        # My M6 term above was for C10? No.
        # M6 contributes to C00 (+1) and C01 (-1).
        
        # Correct M6:
        # U: A10 - A00 -> [u10, u00] with [1, -1]
        # V: B00 + B01 -> [v00, v01]
        # W: C00 (+1), C01 (-1) -> [w00, w01] with [1, -1]
        
        # I will re-implement the Strassen part carefully in the code.
        
        # M7 = (A01-A11)(B10+B11)
        # C00 gets -M7, C10 gets -M7
        # U: A01 - A11 -> [u01, u11] with [1, -1]
        # V: B10 + B11 -> [v10, v11]
        # W: C00 (-1), C10 (-1) -> [w00, w10] with [-1, -1]
        
        # I will rewrite the Strassen addition logic inside the function to be safe.
        
    # Iterate L and R
    for side in [0, 1]:
        j_offset = side * 2 # 0 or 2
        
        # Part 1: A row 0 (1x2) with B (2x6)
        # A row 0: i=0, j=j_offset, j_offset+1
        # B rows: j_offset, j_offset+1
        # Split B cols into 3 blocks of 2
        for k_block in range(3):
            k_offset = k_block * 2
            # 1x2x2 Schoolbook
            # 4 terms
            for j_idx in range(2):
                for k_idx in range(2):
                    i = 0
                    j = j_offset + j_idx
                    k = k_offset + k_idx
                    add_term(i, j, k, 1)
            
        # Part 2: A rows 1,2 (2x2) with B (2x6)
        # A rows: 1, 2. Cols j_offset, j_offset+1
        # B rows: j_offset, j_offset+1
        for k_block in range(3):
            k_offset = k_block * 2
            # 2x2x2 Strassen
            i0 = 1
            j0 = j_offset
            k0 = k_offset
            # Call Strassen helper
            
            # M1
            U[1*m + j0][col] = 1; U[2*m + j0+1][col] = 1
            V[j0*p + k0][col] = 1; V[j0+1*p + k0+1][col] = 1
            W[k0*n + 1][col] = 1; W[k0+1*n + 2][col] = 1 # C00, C11
            col += 1
            
            # M2
            U[1*m + j0+1][col] = 1; U[2*m + j0+1][col] = 1 # A10+A11 (rel i0=1, i1=2)
            # A indices: i0=1, i1=2. j0, j1.
            # A10 is i1,j0? No. A_{i,j}.
            # Local A:
            # 0 0 -> i0, j0
            # 0 1 -> i0, j1
            # 1 0 -> i1, j0
            # 1 1 -> i1, j1
            # M2 = (A10+A11)B00.
            # A10 is (i1, j0). A11 is (i1, j1).
            # So U: (i1, j0) + (i1, j1).
            U[2*m + j0][col] = 1; U[2*m + j0+1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + 2][col] = 1; W[k0+1*n + 2][col] = 1 # C10, C11 (i1=2)
            col += 1
            
            # M3
            U[1*m + j0][col] = 1 # A00
            V[j0*p + k0+1][col] = 1; V[j0+1*p + k0+1][col] = -1 # B01 - B11
            W[k0+1*n + 1][col] = 1; W[k0+1*n + 2][col] = -1 # C01, C11
            col += 1
            
            # M4
            U[2*m + j0+1][col] = 1 # A11
            V[j0+1*p + k0][col] = 1; V[j0*p + k0][col] = -1 # B10 - B00
            W[k0*n + 2][col] = 1; W[k0*n + 1][col] = -1 # C10, C00
            col += 1
            
            # M5
            U[1*m + j0][col] = 1; U[1*m + j0+1][col] = 1 # A00+A01
            V[j0+1*p + k0+1][col] = 1 # B11
            W[k0+1*n + 1][col] = 1; W[k0+1*n + 2][col] = -1 # C01, C11
            col += 1
            
            # M6
            U[2*m + j0][col] = 1; U[1*m + j0][col] = -1 # A10 - A00
            V[j0*p + k0][col] = 1; V[j0*p + k0+1][col] = 1 # B00 + B01
            W[k0*n + 1][col] = 1; W[k0+1*n + 1][col] = -1 # C00, C01
            col += 1
            
            # M7
            U[1*m + j0+1][col] = 1; U[2*m + j0+1][col] = -1 # A01 - A11
            V[j0+1*p + k0][col] = 1; V[j0+1*p + k0+1][col] = 1 # B10 + B11
            W[k0*n + 1][col] = -1; W[k0*n + 2][col] = -1 # C00, C10
            
    return {"U": U, "V": V, "W": W}

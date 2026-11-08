DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    # Helper to add a 2x2x2 Strassen block
    # i_offset: row offset in A (0 or 1 for the 2x2 block within the 2x4 A)
    # j_offset: col offset in A (0 or 2)
    # k_offset: col offset in B (0, 2, 4, or 6)
    # The block covers A[i_offset:i_offset+2, j_offset:j_offset+2]
    # and B[j_offset:j_offset+2, k_offset:k_offset+2]
    # and contributes to C[i_offset:i_offset+2, k_offset:k_offset+2]
    
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # Strassen terms M1..M7
        # x is A, y is B, z is C
        # Local indices 0,1 for rows/cols of the 2x2 blocks
        
        # M1 = (x00 + x11)(y00 + y11)
        # x00: A[i_off, j_off]
        # x11: A[i_off+1, j_off+1]
        # y00: B[j_off, k_off]
        # y11: B[j_off+1, k_off+1]
        
        terms = [
            # (coeff_x00, coeff_x01, coeff_x10, coeff_x11, coeff_y00, coeff_y01, coeff_y10, coeff_y11)
            (1, 0, 0, 1, 1, 0, 0, 1), # M1
            (0, 0, 1, 1, 1, 0, 0, 0), # M2: (x10+x11)y00
            (1, 0, 0, 0, 0, 1, 0, -1), # M3: x00(y01-y11)
            (0, 0, 0, 1, -1, 0, 1, 0), # M4: x11(y10-y00)
            (1, 1, 0, 0, 0, 0, 0, 1), # M5: (x00+x01)y11
            (-1, 0, 1, 0, 1, 1, 0, 0), # M6: (x10-x00)(y00+y01)
            (0, 1, 0, -1, 0, 0, 1, 1)  # M7: (x01-x11)(y10+y11)
        ]
        
        # Reconstruction coefficients for z00, z01, z10, z11
        # z00 = M1 + M4 - M5 + M7
        # z01 = M3 + M5
        # z10 = M2 + M4
        # z11 = M1 - M2 + M3 + M6
        
        recon = [
            # (coeff_M1, coeff_M2, coeff_M3, coeff_M4, coeff_M5, coeff_M6, coeff_M7)
            (1, 0, 0, 1, -1, 0, 1), # z00
            (0, 0, 1, 0, 1, 0, 0),  # z01
            (0, 1, 0, 1, 0, 0, 0),  # z10
            (1, -1, 1, 0, 0, 1, 0)  # z11
        ]
        
        for t in range(7):
            col = col_start + t
            cx = terms[t]
            
            # U part: A indices
            U[i_off * m + j_off][col] += cx[0]
            U[i_off * m + j_off + 1][col] += cx[1]
            U[(i_off + 1) * m + j_off][col] += cx[2]
            U[(i_off + 1) * m + j_off + 1][col] += cx[3]
            
            # V part: B indices
            V[j_off * p + k_off][col] += cx[4]
            V[j_off * p + k_off + 1][col] += cx[5]
            V[(j_off + 1) * p + k_off][col] += cx[6]
            V[(j_off + 1) * p + k_off + 1][col] += cx[7]
            
            # W part: C indices
            # z00 -> C[i_off, k_off]
            W[k_off * n + i_off][col] += recon[0][t]
            # z01 -> C[i_off, k_off+1]
            W[(k_off + 1) * n + i_off][col] += recon[1][t]
            # z10 -> C[i_off+1, k_off]
            W[k_off * n + (i_off + 1)][col] += recon[2][t]
            # z11 -> C[i_off+1, k_off+1]
            W[(k_off + 1) * n + (i_off + 1)][col] += recon[3][t]

    def add_trivial_block(i_off, j_off, k_off, col_start):
        # 2x2x1 block
        # A is 2x2, B is 2x1, C is 2x1
        # Terms: x00*y00, x01*y10, x10*y00, x11*y10
        # z00 = x00*y00 + x01*y10
        # z10 = x10*y00 + x11*y10
        
        # Note: B is 2x1 here, so j_off is 0 or 1 relative to the 2x2 block?
        # Actually, in our split, the last block is 2x1.
        # A is 2x2 (j_off, j_off+1).
        # B is 2x1 (j_off, j_off+1) for rows, and 1 col (k_off).
        
        terms = [
            # x00*y00
            (1, 0, 0, 0, 1, 0, 0, 0),
            # x01*y10
            (0, 1, 0, 0, 0, 0, 1, 0),
            # x10*y00
            (0, 0, 1, 0, 1, 0, 0, 0),
            # x11*y10
            (0, 0, 0, 1, 0, 0, 1, 0)
        ]
        
        recon = [
            # z00 = M1 + M2
            (1, 1, 0, 0),
            # z10 = M3 + M4
            (0, 0, 1, 1)
        ]
        
        for t in range(4):
            col = col_start + t
            cx = terms[t]
            
            # U
            U[i_off * m + j_off][col] += cx[0]
            U[i_off * m + j_off + 1][col] += cx[1]
            U[(i_off + 1) * m + j_off][col] += cx[2]
            U[(i_off + 1) * m + j_off + 1][col] += cx[3]
            
            # V
            # B rows j_off, j_off+1. Col k_off.
            V[j_off * p + k_off][col] += cx[4]
            V[(j_off + 1) * p + k_off][col] += cx[6] # Note: cx[5] and cx[7] are 0 for y01, y11 as there is only 1 col
            
            # W
            # z00 -> C[i_off, k_off]
            W[k_off * n + i_off][col] += recon[0][t]
            # z10 -> C[i_off+1, k_off]
            W[k_off * n + (i_off + 1)][col] += recon[1][t]

    # Process A_L (i_off=0, j_off=0) with B_T (j_off=0)
    # k splits: 0-1 (Strassen), 2-3 (Strassen), 4-5 (Strassen), 6 (Trivial)
    add_strassen_block(0, 0, 0, 0)   # Rank 7
    add_strassen_block(0, 0, 2, 7)   # Rank 7
    add_strassen_block(0, 0, 4, 14)  # Rank 7
    add_trivial_block(0, 0, 6, 21)   # Rank 4
    
    # Process A_R (i_off=0, j_off=2) with B_B (j_off=2)
    # Note: i_off is still 0 for A rows, but j_off is 2 for A cols.
    # Wait, A is 2x4. i is 0,1. j is 0,1,2,3.
    # A_L uses j=0,1. A_R uses j=2,3.
    # So for A_R, the local j indices 0,1 map to global j 2,3.
    # B_B uses j=2,3. Local j 0,1 map to global j 2,3.
    # So j_off for B is 2.
    # k splits are the same.
    
    add_strassen_block(0, 2, 0, 25)  # Rank 7
    add_strassen_block(0, 2, 2, 32)  # Rank 7
    add_strassen_block(0, 2, 4, 39)  # Rank 7
    add_trivial_block(0, 2, 6, 46)   # Rank 4
    
    return {"U": U, "V": V, "W": W}

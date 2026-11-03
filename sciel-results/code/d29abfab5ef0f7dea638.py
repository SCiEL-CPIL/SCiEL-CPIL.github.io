"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition:
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # Split B1, B2 into 3 blocks of 2x2 each.
    # Total 6 blocks of 2x2*2x2.
    # Use Strassen (rank 7) for each block.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2*2x2
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    
    # Reconstruction:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # Define local indices for 2x2 block
    # x indices: (0,0), (0,1), (1,0), (1,1)
    # y indices: (0,0), (0,1), (1,0), (1,1)
    # z indices: (0,0), (0,1), (1,0), (1,1)
    
    # Coefficients for U (X part) for each M
    # U_local[m][i][j]
    U_coeffs = [
        [[1, 0], [0, 1]],  # M1: x00 + x11
        [[0, 0], [1, 1]],  # M2: x10 + x11
        [[1, 0], [0, 0]],  # M3: x00
        [[0, 0], [0, 1]],  # M4: x11
        [[1, 1], [0, 0]],  # M5: x00 + x01
        [[-1, 0], [1, 0]], # M6: x10 - x00
        [[0, 1], [0, -1]], # M7: x01 - x11
    ]
    
    # Coefficients for V (Y part) for each M
    # V_local[m][r][s]
    V_coeffs = [
        [[1, 0], [0, 1]],  # M1: y00 + y11
        [[1, 0], [0, 0]],  # M2: y00
        [[0, 1], [0, -1]], # M3: y01 - y11
        [[-1, 0], [1, 0]], # M4: y10 - y00
        [[0, 0], [0, 1]],  # M5: y11
        [[1, 1], [0, 0]],  # M6: y00 + y01
        [[0, 1], [1, 1]],  # M7: y10 + y11
    ]
    
    # Coefficients for W (Z part) for each M
    # W_local[m][k][i] -> z_ki = sum M * coeff
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    # Note: W index is [k*n + i]. For 2x2 block, k in {0,1}, i in {0,1}.
    # W_local[m][k][i]
    W_coeffs = [
        [[1, 0], [0, 1]], # M1: z00 += 1, z11 += 1. (k=0,i=0)=1, (k=1,i=1)=1
        [[0, 0], [1, -1]], # M2: z10 += 1, z11 -= 1. (k=1,i=0)=1, (k=1,i=1)=-1
        [[0, 1], [0, 1]], # M3: z01 += 1, z11 += 1. (k=0,i=1)=1, (k=1,i=1)=1
        [[1, 0], [1, 0]], # M4: z00 += 1, z10 += 1. (k=0,i=0)=1, (k=1,i=0)=1
        [[-1, 0], [0, 1]], # M5: z00 -= 1, z01 += 1. (k=0,i=0)=-1, (k=0,i=1)=1
        [[0, 0], [0, 1]], # M6: z11 += 1. (k=1,i=1)=1
        [[1, 0], [0, 0]], # M7: z00 += 1. (k=0,i=0)=1
    ]
    
    # Check W_coeffs against reconstruction:
    # z00: M1(1) + M4(1) - M5(-1) + M7(1) -> 1+1+1+1? No.
    # Formula: z00 = M1 + M4 - M5 + M7.
    # My W_coeffs for M5 has -1 at z00. So sum is 1 + 1 + (-1)*(-1)? No.
    # The term is sum_t W_t * M_t.
    # So if W_M5_z00 = -1, then contribution is -1 * M5. Correct.
    
    col = 0
    # Blocks: a in {0, 1} (A block index), b in {0, 1, 2} (B column pair index)
    for a in range(2):
        for b in range(3):
            # Determine global indices for A block
            # A is 2x4. Rows 0,1. Cols 0,1 for a=0; 2,3 for a=1.
            a_row_offset = 0
            a_col_offset = a * 2
            
            # Determine global indices for B block
            # B is 4x6. Rows 0,1 for a=0 (since B1 corresponds to A1); 2,3 for a=1.
            # Wait, A1 multiplies B1 (top half of B). A2 multiplies B2 (bottom half of B).
            # So if a=0, B rows are 0,1. If a=1, B rows are 2,3.
            b_row_offset = a * 2
            # Cols: b=0 -> 0,1; b=1 -> 2,3; b=2 -> 4,5.
            b_col_offset = b * 2
            
            # Determine global indices for C block
            # C is 2x6. Rows 0,1. Cols same as B cols.
            c_row_offset = b_col_offset # C col offset
            c_col_offset = 0 # C row offset (i index)
            
            for t in range(7):
                # Fill U
                for ui in range(2):
                    for uj in range(2):
                        val = U_coeffs[t][ui][uj]
                        if val != 0:
                            global_i = ui
                            global_j = a_col_offset + uj
                            U[global_i * m + global_j][col] += val
                
                # Fill V
                for vr in range(2):
                    for vs in range(2):
                        val = V_coeffs[t][vr][vs]
                        if val != 0:
                            global_j = b_row_offset + vr
                            global_k = b_col_offset + vs
                            V[global_j * p + global_k][col] += val
                
                # Fill W
                for wk in range(2):
                    for wi in range(2):
                        val = W_coeffs[t][wk][wi]
                        if val != 0:
                            global_k = c_row_offset + wk
                            global_i = c_col_offset + wi
                            W[global_k * n + global_i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

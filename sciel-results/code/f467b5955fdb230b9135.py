import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Result reconstruction:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (coeffs for A, coeffs for B, coeffs for C)
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({1: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}), # Wait, z10=M2+M4, z11=M1-M2+M3+M6. 
        # Let's map carefully.
        # M1: A{0,3}, B{0,3}, C{0,3}
        # M2: A{1,3}, B{0}, C{1:-1, 2:1} -> z10=M2+M4, z11=M1-M2... so M2 is -1 in z11, +1 in z10.
        # M3: A{0}, B{2,-3}, C{1,3} -> z01=M3+M5, z11=M1-M2+M3+M6. So M3 is +1 in z01, +1 in z11.
        # M4: A{3}, B{1,-0}, C{0,2} -> z00=M1+M4..., z10=M2+M4. So M4 is +1 in z00, +1 in z10.
        # M5: A{0,1}, B{3}, C{0:-1, 1} -> z00=M1+M4-M5+M7, z01=M3+M5. So M5 is -1 in z00, +1 in z01.
        # M6: A{1,-0}, B{0,1}, C{3} -> z11=...+M6. So M6 is +1 in z11.
        # M7: A{1,-3}, B{2,3}, C{0} -> z00=...+M7. So M7 is +1 in z00.
        
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({1: 1, 3: 1}, {0: 1}, {1: -1, 2: 1}),
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # We have two main blocks: A_L B_T and A_R B_B
    # A_L is A[0:2, 0:2], A_R is A[0:2, 2:4]
    # B_T is B[0:2, :], B_B is B[2:4, :]
    
    # Block 1: A_L B_T
    # A_L rows 0,1. Cols 0,1.
    # B_T rows 0,1. Cols 0,1,2,3,4,5.
    # B_T split into Y1 (cols 0,1), Y2 (cols 2,3), Y3 (cols 4,5).
    
    for block_idx in range(2): # 0 for A_L B_T, 1 for A_R B_B
        if block_idx == 0:
            a_row_offset = 0
            a_col_offset = 0
            b_row_offset = 0
        else:
            a_row_offset = 0
            a_col_offset = 2
            b_row_offset = 2
            
        for y_idx in range(3): # 3 2x2 blocks in B
            b_col_offset = y_idx * 2
            
            # Map local 2x2 indices to global
            # A local (r, c) -> Global (a_row_offset + r, a_col_offset + c)
            # U index = (a_row_offset + r) * m + (a_col_offset + c)
            
            # B local (r, c) -> Global (b_row_offset + r, b_col_offset + c)
            # V index = (b_row_offset + r) * p + (b_col_offset + c)
            
            # C local (r, c) -> Global (r, y_idx * 2 + c)
            # W index = (y_idx * 2 + c) * n + r
            
            for k in range(7):
                col = block_idx * 21 + y_idx * 7 + k
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[k]
                
                for a_loc, val in a_coeffs.items():
                    r, c = divmod(a_loc, 2)
                    u_idx = (a_row_offset + r) * m + (a_col_offset + c)
                    U[u_idx, col] += val
                    
                for b_loc, val in b_coeffs.items():
                    r, c = divmod(b_loc, 2)
                    v_idx = (b_row_offset + r) * p + (b_col_offset + c)
                    V[v_idx, col] += val
                    
                for c_loc, val in c_coeffs.items():
                    r, c = divmod(c_loc, 2)
                    w_idx = (y_idx * 2 + c) * n + r
                    W[w_idx, col] += val
                    
    return {"U": U, "V": V, "W": W}

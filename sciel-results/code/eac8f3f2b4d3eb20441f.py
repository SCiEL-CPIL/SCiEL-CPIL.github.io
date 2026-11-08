import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using 4 Strassen 2x2 multiplications
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # P1..P7
    # U_part (X), V_part (Y), W_part (Z reconstruction)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 0], # P1: X00 + X11 -> indices 0, 3? No, X is 2x2.
        # X00 is index 0, X01 is 1, X10 is 2, X11 is 3 in the 2x2 block?
        # We need to map X indices to global A indices.
    ]
    
    # Let's define the linear forms for X (2x2) and Y (2x2)
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    # P1: (x00+x11)(y00+y11)
    # P2: (x10+x11)y00
    # P3: x00(y01-y11)
    # P4: x11(y10-y00)
    # P5: (x00+x01)y11
    # P6: (x10-x00)(y00+y01)
    # P7: (x01-x11)(y10+y11)
    
    U_forms = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [1, 0, -1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    V_forms = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # Reconstruction coefficients for Z
    # Z00, Z01, Z10, Z11
    W_forms = [
        [1, 0, 0, 0], # P1
        [0, 0, 0, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    # Check P7: Z00 += P7, Z11 += 0?
    # Z11 = P1 - P2 + P3 + P6. P7 is not in Z11.
    # Z00 = P1 + P4 - P5 + P7.
    # So P7 contributes to Z00 only.
    # My W_forms for P7 is [1, 0, 0, 0]. Correct.
    
    # Define the 4 blocks
    # Block 1: A_L (cols 0,1), B_TL (rows 0,1 cols 0,1) -> M1 -> C_L (cols 0,1)
    # Block 2: A_R (cols 2,3), B_BL (rows 2,3 cols 0,1) -> M2 -> C_L (cols 0,1)
    # Block 3: A_L (cols 0,1), B_TR (rows 0,1 cols 2,3) -> M3 -> C_R (cols 2,3)
    # Block 4: A_R (cols 2,3), B_BR (rows 2,3 cols 2,3) -> M4 -> C_R (cols 2,3)
    
    blocks = [
        {
            'A_cols': [0, 1], # A_L
            'B_rows': [0, 1], # B_TL rows
            'B_cols': [0, 1], # B_TL cols
            'C_cols': [0, 1], # Output to C_L
            'base': 0
        },
        {
            'A_cols': [2, 3], # A_R
            'B_rows': [2, 3], # B_BL rows
            'B_cols': [0, 1], # B_BL cols
            'C_cols': [0, 1], # Output to C_L
            'base': 7
        },
        {
            'A_cols': [0, 1], # A_L
            'B_rows': [0, 1], # B_TR rows
            'B_cols': [2, 3], # B_TR cols
            'C_cols': [2, 3], # Output to C_R
            'base': 14
        },
        {
            'A_cols': [2, 3], # A_R
            'B_rows': [2, 3], # B_BR rows
            'B_cols': [2, 3], # B_BR cols
            'C_cols': [2, 3], # Output to C_R
            'base': 21
        }
    ]
    
    for b in blocks:
        a_cols = b['A_cols']
        b_rows = b['B_rows']
        b_cols = b['C_cols'] # Wait, b_cols in block dict is B cols. Let's rename.
        b_cols_b = b['B_cols']
        c_cols = b['C_cols']
        base = b['base']
        
        for k in range(7):
            col = base + k
            
            # Fill U
            u_form = U_forms[k]
            # X00 -> A[0, a_cols[0]]
            # X01 -> A[0, a_cols[1]]
            # X10 -> A[1, a_cols[0]]
            # X11 -> A[1, a_cols[1]]
            
            # Global index in U: i * m + j
            # i is row of A (0 or 1)
            # j is col of A
            
            # u_form[0] is coeff for X00
            if u_form[0] != 0:
                idx = 0 * m + a_cols[0]
                U[idx][col] += u_form[0]
            if u_form[1] != 0:
                idx = 0 * m + a_cols[1]
                U[idx][col] += u_form[1]
            if u_form[2] != 0:
                idx = 1 * m + a_cols[0]
                U[idx][col] += u_form[2]
            if u_form[3] != 0:
                idx = 1 * m + a_cols[1]
                U[idx][col] += u_form[3]
                
            # Fill V
            v_form = V_forms[k]
            # Y00 -> B[b_rows[0], b_cols_b[0]]
            # Y01 -> B[b_rows[0], b_cols_b[1]]
            # Y10 -> B[b_rows[1], b_cols_b[0]]
            # Y11 -> B[b_rows[1], b_cols_b[1]]
            
            # Global index in V: j * p + k'
            # j is row of B
            # k' is col of B
            
            if v_form[0] != 0:
                idx = b_rows[0] * p + b_cols_b[0]
                V[idx][col] += v_form[0]
            if v_form[1] != 0:
                idx = b_rows[0] * p + b_cols_b[1]
                V[idx][col] += v_form[1]
            if v_form[2] != 0:
                idx = b_rows[1] * p + b_cols_b[0]
                V[idx][col] += v_form[2]
            if v_form[3] != 0:
                idx = b_rows[1] * p + b_cols_b[1]
                V[idx][col] += v_form[3]
                
            # Fill W
            w_form = W_forms[k]
            # Z00 -> C[0, c_cols[0]]
            # Z01 -> C[0, c_cols[1]]
            # Z10 -> C[1, c_cols[0]]
            # Z11 -> C[1, c_cols[1]]
            
            # Global index in W: k' * n + i
            # k' is col of C
            # i is row of C
            
            if w_form[0] != 0:
                idx = c_cols[0] * n + 0
                W[idx][col] += w_form[0]
            if w_form[1] != 0:
                idx = c_cols[1] * n + 0
                W[idx][col] += w_form[1]
            if w_form[2] != 0:
                idx = c_cols[0] * n + 1
                W[idx][col] += w_form[2]
            if w_form[3] != 0:
                idx = c_cols[1] * n + 1
                W[idx][col] += w_form[3]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

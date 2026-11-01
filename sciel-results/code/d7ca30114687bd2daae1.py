import numpy as np

def construct():
    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n
    
    U = []
    V = []
    W = []
    
    # Strassen formulas for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y21+Y22)
    # M4 = X22(Y11-Y21)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Indices for X (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for Y (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for Z (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1], # X11+X22
        [0, 0, 1, 1], # X21+X22
        [1, 0, 0, 0], # X11
        [0, 0, 0, 1], # X22
        [1, 1, 0, 0], # X11+X12
        [-1, 0, 1, 0], # X21-X11
        [0, 1, 0, -1]  # X12-X22
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # Y11+Y22
        [1, 0, 0, 0], # Y11
        [0, 0, 1, 1], # Y21+Y22
        [1, -1, 0, 0], # Y11-Y21
        [0, 0, 0, 1], # Y22
        [1, 1, 0, 0], # Y11+Y12
        [0, 0, 1, 1]  # Y21+Y22
    ]
    
    strassen_W = [
        [1, 0, 0, 0], # M1 -> Z11
        [0, 1, 0, 0], # M3+M5 -> Z12 (M3 is 0, M5 is 1? No, let's map carefully)
        [0, 0, 1, 0], # M2+M4 -> Z21
        [0, 0, 0, 1]  # M1-M2+M3+M6 -> Z22
    ]
    
    # Let's explicitly define the linear combinations for W
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We will add 7 columns per Strassen block.
    # For each term t in 0..6:
    # U_col = strassen_U[t] mapped to global A indices
    # V_col = strassen_V[t] mapped to global B indices
    # W_col = sum of coefficients for each Z entry mapped to global C indices
    
    def add_strassen_block(a_offset, b_offset, c_offset):
        # a_offset: index of top-left of A block (0 or 4)
        # b_offset: index of top-left of B block (2c or 6+2c? No, B is 4x6)
        # b_offset is index in V (4x6) of top-left of B block.
        # c_offset: index of top-left of C block in W (6x2)
        
        for t in range(7):
            u_col = np.zeros(nm)
            v_col = np.zeros(mp)
            w_col = np.zeros(pn)
            
            # Fill U
            for idx, val in enumerate(strassen_U[t]):
                if val != 0:
                    u_col[a_offset + idx] = val
            
            # Fill V
            # B block is 2x2. Indices in V (row-major 4x6):
            # (0,0) -> 0, (0,1) -> 1
            # (1,0) -> 6, (1,1) -> 7
            # So b_offset is the index of (0,0).
            # (0,0) is b_offset
            # (0,1) is b_offset + 1
            # (1,0) is b_offset + 6
            # (1,1) is b_offset + 7
            for idx, val in enumerate(strassen_V[t]):
                if val != 0:
                    if idx == 0: v_col[b_offset] = val
                    elif idx == 1: v_col[b_offset + 1] = val
                    elif idx == 2: v_col[b_offset + 6] = val
                    elif idx == 3: v_col[b_offset + 7] = val
            
            # Fill W
            # C block is 2x2. Indices in W (row-major 6x2):
            # Output Z is 2x2.
            # Z(0,0) -> C(0, 2c) -> W index: (2c)*2 + 0 = 4c
            # Z(0,1) -> C(0, 2c+1) -> W index: (2c+1)*2 + 0 = 4c+2
            # Z(1,0) -> C(1, 2c) -> W index: (2c)*2 + 1 = 4c+1
            # Z(1,1) -> C(1, 2c+1) -> W index: (2c+1)*2 + 1 = 4c+3
            
            # c_offset is 4c.
            # Z(0,0) is at c_offset
            # Z(1,0) is at c_offset + 1
            # Z(0,1) is at c_offset + 2
            # Z(1,1) is at c_offset + 3
            
            # Coefficients for each Z entry from M1..M7
            # Z11: M1, M4, -M5, M7
            # Z12: M3, M5
            # Z21: M2, M4
            # Z22: M1, -M2, M3, M6
            
            w_coeffs = np.zeros(4)
            
            if t == 0: # M1
                w_coeffs[0] += 1
                w_coeffs[3] += 1
            elif t == 1: # M2
                w_coeffs[2] += 1
                w_coeffs[3] -= 1
            elif t == 2: # M3
                w_coeffs[1] += 1
                w_coeffs[3] += 1
            elif t == 3: # M4
                w_coeffs[0] += 1
                w_coeffs[2] += 1
            elif t == 4: # M5
                w_coeffs[0] -= 1
                w_coeffs[1] += 1
            elif t == 5: # M6
                w_coeffs[3] += 1
            elif t == 6: # M7
                w_coeffs[0] += 1
            
            for z_idx, val in enumerate(w_coeffs):
                if val != 0:
                    w_col[c_offset + z_idx] = val
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)

    # 3 column blocks for C
    for c in range(3):
        # Block 1: A_L * B_{c,T}
        # A_L is cols 0,1 of A. Top-left index 0.
        # B_{c,T} is rows 0,1 and cols 2c, 2c+1 of B. Top-left index 2c.
        add_strassen_block(0, 2*c, 4*c)
        
        # Block 2: A_R * B_{c,B}
        # A_R is cols 2,3 of A. Top-left index 4.
        # B_{c,B} is rows 2,3 and cols 2c, 2c+1 of B. Top-left index 12+2c.
        add_strassen_block(4, 12+2*c, 4*c)

    return {
        "U": np.array(U).T.tolist(),
        "V": np.array(V).T.tolist(),
        "W": np.array(W).T.tolist()
    }

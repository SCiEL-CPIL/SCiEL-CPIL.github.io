import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen forms for 2x2 multiplication
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2/V2/W2 correspond to a,b,c,d / e,f,g,h / C00,C01,C10,C11
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # S matrix: 4x7, rows are C00, C01, C10, C11 blocks
    S = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Define X_k and Y_k linear forms
    # X_k is 2x2, entries are linear forms in A
    # A indices: 0..3
    # Block (r,c) of A: rows 2r, 2r+1; cols 2c, 2c+1
    # A[r,c][i,j] = A[2r+i, 2c+j]
    
    # X_k definitions (as lists of 4 linear forms, each a dict {index: coeff})
    # Or just lists of coefficients for A[0..15]
    # Let's use a helper to get the linear form for a block entry
    def get_block_entry_linear_form(r, c, i, j):
        # Returns list of 16 coeffs
        coeffs = [0] * 16
        idx = (2 * r + i) * 4 + (2 * c + j)
        coeffs[idx] = 1
        return coeffs

    # X_k: list of 7 items, each is 4 linear forms (length 16)
    # X_1 = A00 + A11
    # X_2 = A10 + A11
    # X_3 = A00
    # X_4 = A11
    # X_5 = A00 + A01
    # X_6 = A10 - A00
    # X_7 = A01 - A11
    
    X_forms = []
    for k in range(7):
        Xk = []
        for i in range(2):
            for j in range(2):
                lf = [0] * 16
                if k == 0: # A00 + A11
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                elif k == 1: # A10 + A11
                    lf[(2*1+i)*4 + (2*0+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                elif k == 2: # A00
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                elif k == 3: # A11
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                elif k == 4: # A00 + A01
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                    lf[(2*0+i)*4 + (2*1+j)] = 1
                elif k == 5: # A10 - A00
                    lf[(2*1+i)*4 + (2*0+j)] = 1
                    lf[(2*0+i)*4 + (2*0+j)] = -1
                elif k == 6: # A01 - A11
                    lf[(2*0+i)*4 + (2*1+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = -1
                Xk.append(lf)
        X_forms.append(Xk)
        
    # Y_k similar
    Y_forms = []
    for k in range(7):
        Yk = []
        for i in range(2):
            for j in range(2):
                lf = [0] * 16
                if k == 0: # B00 + B11
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                elif k == 1: # B00
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                elif k == 2: # B01 - B11
                    lf[(2*0+i)*4 + (2*1+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = -1
                elif k == 3: # B10 - B00
                    lf[(2*1+i)*4 + (2*0+j)] = 1
                    lf[(2*0+i)*4 + (2*0+j)] = -1
                elif k == 4: # B11
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                elif k == 5: # B00 + B01
                    lf[(2*0+i)*4 + (2*0+j)] = 1
                    lf[(2*0+i)*4 + (2*1+j)] = 1
                elif k == 6: # B10 + B11
                    lf[(2*1+i)*4 + (2*0+j)] = 1
                    lf[(2*1+i)*4 + (2*1+j)] = 1
                Yk.append(lf)
        Y_forms.append(Yk)

    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for k in range(7):
        for t in range(7):
            col = 7 * k + t
            
            # U part
            # u_t is U2[:, t]
            # X_k entries are X_forms[k][0..3]
            # U[i, col] = sum_{r,c} U2[r*2+c, t] * X_forms[k][r*2+c][i]
            for i in range(16):
                val = 0
                for r in range(2):
                    for c in range(2):
                        idx_2 = r * 2 + c
                        coeff = U2[idx_2][t]
                        if coeff != 0:
                            val += coeff * X_forms[k][idx_2][i]
                U[i][col] = val
                
            # V part
            for i in range(16):
                val = 0
                for r in range(2):
                    for c in range(2):
                        idx_2 = r * 2 + c
                        coeff = V2[idx_2][t]
                        if coeff != 0:
                            val += coeff * Y_forms[k][idx_2][i]
                V[i][col] = val
                
            # W part
            # w_t is W2[:, t]
            # For each global C index (i, j), determine block (bi, bj) and local (li, lj)
            # W[i, col] = S[bi*2+bj, k] * W2[li*2+lj, t]
            for i in range(16):
                bi = i // 4
                bj = i % 4
                li = i % 4 # Wait, i is flattened 4x4.
                # i = row * 4 + col
                row = i // 4
                col_idx = i % 4
                bi = row // 2
                bj = col_idx // 2
                li = row % 2
                lj = col_idx % 2
                
                block_idx = bi * 2 + bj
                local_idx = li * 2 + lj
                
                coeff_S = S[block_idx][k]
                coeff_W2 = W2[local_idx][t]
                
                W[i][col] = coeff_S * coeff_W2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

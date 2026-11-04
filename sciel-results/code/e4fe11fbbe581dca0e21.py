import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    # Initialize matrices
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get A index
    def a_idx(i, j):
        return i * m + j
        
    # Helper to get B index
    def b_idx(j, k):
        return j * p + k
        
    # Helper to get W index (C index)
    # W[k*n + i, r] corresponds to C[i, k]
    def w_idx(i, k):
        return k * n + i

    # Strassen coefficients for 2x2 multiplication
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y01 - y11)
    # m4 = x11(y10 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    
    # Z = XY
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Coefficients c[s, r, c] for Z[r, c] from m_s
    # s in 0..6, r in 0..1, c in 0..1
    c_coeffs = {
        0: {(0,0): 1, (1,1): 1},  # m1
        1: {(1,0): 1, (1,1): -1}, # m2
        2: {(0,1): 1, (1,1): 1},  # m3
        3: {(0,0): 1, (1,0): 1},  # m4
        4: {(0,1): 1, (1,1): -1}, # m5
        5: {(0,0): 1, (1,0): 1},  # m6
        6: {(0,1): 1, (1,1): -1}  # m7
    }
    
    # Wait, the c_coeffs above are for the "sums" inside m_s?
    # No, m_s is a product of two sums.
    # We need to expand each m_s into products of individual A and B elements.
    # This is getting complicated to do symbolically in code.
    
    # Alternative: Hardcode the 49 terms?
    # Or use a simpler decomposition?
    
    # Let's stick to the recursive Strassen logic but implement it by iterating over the scalar elements.
    
    # We define the 7 Strassen terms for the 4x4 case.
    # Each term M_t is a product of two 2x2 matrices X_t and Y_t.
    # X_t and Y_t are linear combinations of the 4 blocks of A and B.
    
    # Let's define the blocks of A:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Same for B.
    
    # We will define the linear combinations for X_t and Y_t.
    # Each entry of X_t is a list of (coeff, a_row, a_col).
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # X1:
    # x00 = A00[0,0] + A11[0,0]
    # x01 = A00[0,1] + A11[0,1]
    # x10 = A00[1,0] + A11[1,0]
    # x11 = A00[1,1] + A11[1,1]
    # Y1:
    # y00 = B00[0,0] + B11[0,0]
    # ...
    
    # This is too verbose. Let's use a more compact representation.
    
    # We'll create a list of 49 tuples: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {a_idx: coeff}
    # V_coeffs: dict {b_idx: coeff}
    # W_coeffs: dict {w_idx: coeff}
    
    terms = []
    
    # Define block getters
    def get_block_A(br, bc):
        # Returns list of 4 (coeff, i, j) for the block
        # Block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
        res = []
        for r in range(2):
            for c in range(2):
                i = 2*br + r
                j = 2*bc + c
                res.append((1, i, j))
        return res

    def get_block_B(br, bc):
        # Block (br, bc) of B. B is m x p.
        # Rows 2*br..2*br+1, Cols 2*bc..2*bc+1
        res = []
        for r in range(2):
            for c in range(2):
                j = 2*br + r
                k = 2*bc + c
                res.append((1, j, k))
        return res

    # Define the 7 Strassen terms
    # Each term is (X_def, Y_def, D_def)
    # X_def: list of 4 (list of (coeff, block_A)) for x00, x01, x10, x11
    # Y_def: list of 4 (list of (coeff, block_B)) for y00, y01, y10, y11
    # D_def: 2x2 matrix of coefficients for C blocks
    
    # M1 = (A00 + A11)(B00 + B11)
    X1 = [
        [(1, (0,0)), (1, (1,1))], # x00
        [(1, (0,0)), (1, (1,1))], # x01
        [(1, (0,0)), (1, (1,1))], # x10
        [(1, (0,0)), (1, (1,1))]  # x11
    ]
    # Wait, x00 is A00[0,0] + A11[0,0].
    # My X1 definition above says x00 is sum of block (0,0) and (1,1) with coeff 1.
    # But block (0,0) has 4 elements. x00 should only take the [0,0] element of those blocks.
    # So X1[0] should be [(1, (0,0, 0,0)), (1, (1,1, 0,0))].
    # I need to specify the element within the block.
    
    # Let's refine the structure.
    # An "element" is (block_row, block_col, local_row, local_col).
    
    def make_A_elem(br, bc, lr, lc):
        return (br, bc, lr, lc)
        
    def make_B_elem(br, bc, lr, lc):
        return (br, bc, lr, lc)

    # M1 = (A00 + A11)(B00 + B11)
    X1 = [
        [(1, make_A_elem(0,0,0,0)), (1, make_A_elem(1,1,0,0))], # x00
        [(1, make_A_elem(0,0,0,1)), (1, make_A_elem(1,1,0,1))], # x01
        [(1, make_A_elem(0,0,1,0)), (1, make_A_elem(1,1,1,0))], # x10
        [(1, make_A_elem(0,0,1,1)), (1, make_A_elem(1,1,1,1))]  # x11
    ]
    Y1 = [
        [(1, make_B_elem(0,0,0,0)), (1, make_B_elem(1,1,0,0))], # y00
        [(1, make_B_elem(0,0,0,1)), (1, make_B_elem(1,1,0,1))], # y01
        [(1, make_B_elem(0,0,1,0)), (1, make_B_elem(1,1,1,0))], # y10
        [(1, make_B_elem(0,0,1,1)), (1, make_B_elem(1,1,1,1))]  # y11
    ]
    D1 = [[1, 1], [1, 1]] # C00 = M1[0,0], C01 = M1[0,1], C10 = M1[1,0], C11 = M1[1,1]
    # Actually, C = M1 + M4 - M5 + M7 for C00? No.
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,0] + M5[0,0]
    # C10 = M2[0,0] + M4[0,0]
    # C11 = M1[0,0] - M2[0,0] + M3[0,0] + M6[0,0]
    
    # So D_t is a 2x2 matrix where D_t[r,c] is the coefficient of M_t[r,c] in C_block.
    # Wait, M_t is a 2x2 matrix. C is 2x2 blocks.
    # C_block(i,j) = sum_t D_t[i,j] * M_t[i,j]? No.
    # C_block(i,j) is formed by linear combinations of the entries of M_t.
    # Specifically:
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,0] + M5[0,0]  -- Wait, C01 is the top-right block.
    # C01 = M3[0,1] + M5[0,1]? No.
    # The formula for Z=XY gives the 2x2 matrix Z.
    # Z[0,0] is the top-left of the block product.
    # So C_block(0,0) uses Z[0,0] of each M_t.
    # C_block(0,1) uses Z[0,1] of each M_t.
    # C_block(1,0) uses Z[1,0] of each M_t.
    # C_block(1,1) uses Z[1,1] of each M_t.
    
    # So for each M_t, we have 4 entries.
    # Each entry M_t[r,c] is a linear combination of the 7 scalar products m_s.
    # So M_t[r,c] = sum_s coeff_s * m_s.
    
    # We already have c_coeffs[s, r, c] from before?
    # No, c_coeffs was for the expansion of m_s into Z entries.
    # Let's re-verify.
    # m1 contributes to z00 (coeff 1), z11 (coeff 1).
    # m2 contributes to z10 (coeff 1), z11 (coeff -1).
    # m3 contributes to z01 (coeff 1), z11 (coeff 1).
    # m4 contributes to z00 (coeff 1), z10 (coeff 1).
    # m5 contributes to z01 (coeff 1), z11 (coeff -1).
    # m6 contributes to z00 (coeff 1), z10 (coeff 1).
    # m7 contributes to z00 (coeff 1), z11 (coeff -1).
    
    # Let's build the matrix Z_coeffs[s, r, c]
    Z_coeffs = np.zeros((7, 2, 2), dtype=int)
    Z_coeffs[0, 0, 0] += 1; Z_coeffs[0, 1, 1] += 1
    Z_coeffs[1, 1, 0] += 1; Z_coeffs[1, 1, 1] -= 1
    Z_coeffs[2, 0, 1] += 1; Z_coeffs[2, 1, 1] += 1
    Z_coeffs[3, 0, 0] += 1; Z_coeffs[3, 1, 0] += 1
    Z_coeffs[4, 0, 1] += 1; Z_coeffs[4, 1, 1] -= 1
    Z_coeffs[5, 0, 0] += 1; Z_coeffs[5, 1, 0] += 1
    Z_coeffs[6, 0, 0] += 1; Z_coeffs[6, 1, 1] -= 1
    
    # Now, define the 7 terms M_t.
    # For each t, we need X_t and Y_t.
    # And D_t (coefficients for C blocks).
    
    D_coeffs = np.zeros((7, 2, 2), dtype=int)
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    D_coeffs[0, 0, 0] += 1
    D_coeffs[3, 0, 0] += 1
    D_coeffs[4, 0, 0] -= 1
    D_coeffs[6, 0, 0] += 1
    # C01 = M3[0,1] + M5[0,1]
    D_coeffs[2, 0, 1] += 1
    D_coeffs[4, 0, 1] += 1
    # C10 = M2[1,0] + M4[1,0]
    D_coeffs[1, 1, 0] += 1
    D_coeffs[3, 1, 0] += 1
    # C11 = M1[1,1] - M2[1,1] + M3[1,1] + M6[1,1]
    D_coeffs[0, 1, 1] += 1
    D_coeffs[1, 1, 1] -= 1
    D_coeffs[2, 1, 1] += 1
    D_coeffs[5, 1, 1] += 1

    # Now define X_t and Y_t for t=0..6
    # t=0: M1 = (A00 + A11)(B00 + B11)
    # t=1: M2 = (A10 + A11)B00
    # t=2: M3 = A00(B01 - B11)
    # t=3: M4 = A11(B10 - B00)
    # t=4: M5 = (A00 + A01)B11
    # t=5: M6 = (A10 - A00)(B00 + B01)
    # t=6: M7 = (A01 - A11)(B10 + B11)
    
    # We need to generate the list of (coeff, elem) for each x_rc and y_rc.
    
    def build_XY(t):
        X = [[None for _ in range(2)] for _ in range(2)]
        Y = [[None for _ in range(2)] for _ in range(2)]
        
        if t == 0:
            # X = A00 + A11
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(0,0,r,c)), (1, make_A_elem(1,1,r,c))]
                    Y[r][c] = [(1, make_B_elem(0,0,r,c)), (1, make_B_elem(1,1,r,c))]
        elif t == 1:
            # X = A10 + A11
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(1,0,r,c)), (1, make_A_elem(1,1,r,c))]
                    Y[r][c] = [(1, make_B_elem(0,0,r,c))]
        elif t == 2:
            # X = A00
            # Y = B01 - B11
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(0,0,r,c))]
                    Y[r][c] = [(1, make_B_elem(0,1,r,c)), (-1, make_B_elem(1,1,r,c))]
        elif t == 3:
            # X = A11
            # Y = B10 - B00
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(1,1,r,c))]
                    Y[r][c] = [(1, make_B_elem(1,0,r,c)), (-1, make_B_elem(0,0,r,c))]
        elif t == 4:
            # X = A00 + A01
            # Y = B11
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(0,0,r,c)), (1, make_A_elem(0,1,r,c))]
                    Y[r][c] = [(1, make_B_elem(1,1,r,c))]
        elif t == 5:
            # X = A10 - A00
            # Y = B00 + B01
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(1,0,r,c)), (-1, make_A_elem(0,0,r,c))]
                    Y[r][c] = [(1, make_B_elem(0,0,r,c)), (1, make_B_elem(0,1,r,c))]
        elif t == 6:
            # X = A01 - A11
            # Y = B10 + B11
            for r in range(2):
                for c in range(2):
                    X[r][c] = [(1, make_A_elem(0,1,r,c)), (-1, make_A_elem(1,1,r,c))]
                    Y[r][c] = [(1, make_B_elem(1,0,r,c)), (1, make_B_elem(1,1,r,c))]
                    
        return X, Y

    col = 0
    for t in range(7):
        X, Y = build_XY(t)
        for s in range(7):
            # Determine which x and y are used in m_s
            # m1: x00+x11, y00+y11
            # m2: x10+x11, y00
            # m3: x00, y01-y11
            # m4: x11, y10-y00
            # m5: x00+x01, y11
            # m6: x10-x00, y00+y01
            # m7: x01-x11, y10+y11
            
            if s == 0:
                x_terms = X[0][0] + X[1][1]
                y_terms = Y[0][0] + Y[1][1]
            elif s == 1:
                x_terms = X[1][0] + X[1][1]
                y_terms = Y[0][0]
            elif s == 2:
                x_terms = X[0][0]
                y_terms = Y[0][1] + Y[1][1] # Note: Y[1][1] is multiplied by -1 in formula?
                # Formula: m3 = x00(y01 - y11). So y_terms should have -1 for y11.
                # My Y definitions have coeffs.
                # For t=2, Y[0][1] is B01[0,1], Y[1][1] is B11[0,1] with -1.
                # So summing them is correct.
            elif s == 3:
                x_terms = X[1][1]
                y_terms = Y[1][0] + Y[0][0] # y10 - y00
            elif s == 4:
                x_terms = X[0][0] + X[0][1]
                y_terms = Y[1][1]
            elif s == 5:
                x_terms = X[1][0] + X[0][0] # x10 - x00
                y_terms = Y[0][0] + Y[0][1]
            elif s == 6:
                x_terms = X[0][1] + X[1][1] # x01 - x11
                y_terms = Y[1][0] + Y[1][1]
            
            # Accumulate U and V coefficients
            u_dict = {}
            for coeff, elem in x_terms:
                br, bc, lr, lc = elem
                i = 2*br + lr
                j = 2*bc + lc
                idx = i * m + j
                u_dict[idx] = u_dict.get(idx, 0) + coeff
                
            v_dict = {}
            for coeff, elem in y_terms:
                br, bc, lr, lc = elem
                j = 2*br + lr
                k = 2*bc + lc
                idx = j * p + k
                v_dict[idx] = v_dict.get(idx, 0) + coeff
            
            # Accumulate W coefficients
            # m_s contributes to Z[r,c] with Z_coeffs[s, r, c]
            # Z[r,c] contributes to C_block(r,c) with D_coeffs[t, r, c]
            # C_block(r,c) is at global rows 2r..2r+1, cols 2c..2c+1
            w_dict = {}
            for r in range(2):
                for c in range(2):
                    z_coeff = Z_coeffs[s, r, c]
                    d_coeff = D_coeffs[t, r, c]
                    total_coeff = z_coeff * d_coeff
                    if total_coeff != 0:
                        # C_block(r,c) entries
                        for lr in range(2):
                            for lc in range(2):
                                i = 2*r + lr
                                k = 2*c + lc
                                w_idx_val = k * n + i
                                w_dict[w_idx_val] = w_dict.get(w_idx_val, 0) + total_coeff
            
            # Write to U, V, W
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_dict.items():
                if val != 0:
                    W[idx][col] = val
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

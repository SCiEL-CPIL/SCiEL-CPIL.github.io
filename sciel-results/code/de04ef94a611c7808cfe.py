import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 terms
    # Each term is a list of (coeff, x_idx, y_idx, z_idx)
    # x_idx, y_idx are (row, col) in 2x2
    # z_idx is (row, col) in 2x2 result
    # We need to define the 7 multiplications
    
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We will iterate through the 7 4x4-Strassen terms
    # And for each, iterate through the 7 2x2-Strassen terms
    
    # Define the 2x2 Strassen decomposition as a list of 7 items.
    # Each item is a dict or tuple describing the multiplication.
    # For each multiplication, we have:
    # - Coefficients for X (2x2)
    # - Coefficients for Y (2x2)
    # - Which Z component it contributes to, and with what weight.
    
    # Let's define the 2x2 Strassen terms explicitly.
    # term: (x_coeffs, y_coeffs, z_contribs)
    # x_coeffs: dict {(r,c): coeff}
    # y_coeffs: dict {(r,c): coeff}
    # z_contribs: list of (z_r, z_c, coeff)
    
    s2_terms = []
    
    # M1
    x1 = {(0,0): 1, (1,1): 1}
    y1 = {(0,0): 1, (1,1): 1}
    z1 = [(0,0, 1), (1,1, 1)]
    s2_terms.append((x1, y1, z1))
    
    # M2
    x2 = {(1,0): 1, (1,1): 1}
    y2 = {(0,0): 1}
    z2 = [(1,0, 1)]
    s2_terms.append((x2, y2, z2))
    
    # M3
    x3 = {(0,0): 1}
    y3 = {(0,1): 1, (1,1): -1}
    z3 = [(0,1, 1)]
    s2_terms.append((x3, y3, z3))
    
    # M4
    x4 = {(1,1): 1}
    y4 = {(1,0): 1, (0,0): -1}
    z4 = [(0,0, 1), (1,0, 1)]
    s2_terms.append((x4, y4, z4))
    
    # M5
    x5 = {(0,0): 1, (0,1): 1}
    y5 = {(1,1): 1}
    z5 = [(0,0, -1), (0,1, 1)]
    s2_terms.append((x5, y5, z5))
    
    # M6
    x6 = {(1,0): 1, (0,0): -1}
    y6 = {(0,0): 1, (0,1): 1}
    z6 = [(1,1, 1)]
    s2_terms.append((x6, y6, z6))
    
    # M7
    x7 = {(0,1): 1, (1,1): -1}
    y7 = {(1,0): 1, (1,1): 1}
    z7 = [(0,0, 1), (1,1, 1)]
    s2_terms.append((x7, y7, z7))
    
    # Now, the 4x4 Strassen terms
    # Each term k (0..6) corresponds to a multiplication of two 2x2 blocks.
    # We need to map the 2x2 indices to 4x4 indices.
    # For term k, we have X_block and Y_block.
    # X_block is a linear combination of A_blocks.
    # Y_block is a linear combination of B_blocks.
    
    # Define the 7 4x4 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each term, define:
    # x_map: dict {(r,c): list of (A_block, coeff)}
    # y_map: dict {(r,c): list of (B_block, coeff)}
    # z_map: dict {(r,c): list of (C_block, coeff)}
    
    # Blocks are (0,0), (0,1), (1,0), (1,1)
    
    def get_4x4_maps(term_idx):
        if term_idx == 0: # M1
            x_map = {
                (0,0): [('A00', 1), ('A11', 1)],
                (0,1): [('A00', 1), ('A11', 1)],
                (1,0): [('A00', 1), ('A11', 1)],
                (1,1): [('A00', 1), ('A11', 1)]
            }
            y_map = {
                (0,0): [('B00', 1), ('B11', 1)],
                (0,1): [('B00', 1), ('B11', 1)],
                (1,0): [('B00', 1), ('B11', 1)],
                (1,1): [('B00', 1), ('B11', 1)]
            }
            z_map = {
                (0,0): [('C00', 1), ('C11', 1)],
                (0,1): [('C01', 1), ('C11', 1)], # Wait, C11 gets M1?
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # So M1 contributes to C00 and C11.
                # The components of M1 (which is 2x2) add to the corresponding components of C00 and C11.
                # So M1[0,0] adds to C00[0,0] and C11[0,0].
                # M1[0,1] adds to C00[0,1] and C11[0,1].
                # etc.
            }
            # Correct z_map for M1:
            z_map = {
                (0,0): [('C00', 1), ('C11', 1)],
                (0,1): [('C00', 1), ('C11', 1)],
                (1,0): [('C00', 1), ('C11', 1)],
                (1,1): [('C00', 1), ('C11', 1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 1: # M2 = (A10+A11)B00
            x_map = {
                (0,0): [('A10', 1), ('A11', 1)],
                (0,1): [('A10', 1), ('A11', 1)],
                (1,0): [('A10', 1), ('A11', 1)],
                (1,1): [('A10', 1), ('A11', 1)]
            }
            y_map = {
                (0,0): [('B00', 1)],
                (0,1): [('B00', 1)],
                (1,0): [('B00', 1)],
                (1,1): [('B00', 1)]
            }
            z_map = {
                (0,0): [('C10', 1), ('C11', -1)],
                (0,1): [('C10', 1), ('C11', -1)],
                (1,0): [('C10', 1), ('C11', -1)],
                (1,1): [('C10', 1), ('C11', -1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 2: # M3 = A00(B01-B11)
            x_map = {
                (0,0): [('A00', 1)],
                (0,1): [('A00', 1)],
                (1,0): [('A00', 1)],
                (1,1): [('A00', 1)]
            }
            y_map = {
                (0,0): [('B01', 1), ('B11', -1)],
                (0,1): [('B01', 1), ('B11', -1)],
                (1,0): [('B01', 1), ('B11', -1)],
                (1,1): [('B01', 1), ('B11', -1)]
            }
            z_map = {
                (0,0): [('C01', 1), ('C11', 1)],
                (0,1): [('C01', 1), ('C11', 1)],
                (1,0): [('C01', 1), ('C11', 1)],
                (1,1): [('C01', 1), ('C11', 1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 3: # M4 = A11(B10-B00)
            x_map = {
                (0,0): [('A11', 1)],
                (0,1): [('A11', 1)],
                (1,0): [('A11', 1)],
                (1,1): [('A11', 1)]
            }
            y_map = {
                (0,0): [('B10', 1), ('B00', -1)],
                (0,1): [('B10', 1), ('B00', -1)],
                (1,0): [('B10', 1), ('B00', -1)],
                (1,1): [('B10', 1), ('B00', -1)]
            }
            z_map = {
                (0,0): [('C00', 1), ('C10', 1)],
                (0,1): [('C00', 1), ('C10', 1)],
                (1,0): [('C00', 1), ('C10', 1)],
                (1,1): [('C00', 1), ('C10', 1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 4: # M5 = (A00+A01)B11
            x_map = {
                (0,0): [('A00', 1), ('A01', 1)],
                (0,1): [('A00', 1), ('A01', 1)],
                (1,0): [('A00', 1), ('A01', 1)],
                (1,1): [('A00', 1), ('A01', 1)]
            }
            y_map = {
                (0,0): [('B11', 1)],
                (0,1): [('B11', 1)],
                (1,0): [('B11', 1)],
                (1,1): [('B11', 1)]
            }
            z_map = {
                (0,0): [('C00', -1), ('C01', 1)],
                (0,1): [('C00', -1), ('C01', 1)],
                (1,0): [('C00', -1), ('C01', 1)],
                (1,1): [('C00', -1), ('C01', 1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 5: # M6 = (A10-A00)(B00+B01)
            x_map = {
                (0,0): [('A10', 1), ('A00', -1)],
                (0,1): [('A10', 1), ('A00', -1)],
                (1,0): [('A10', 1), ('A00', -1)],
                (1,1): [('A10', 1), ('A00', -1)]
            }
            y_map = {
                (0,0): [('B00', 1), ('B01', 1)],
                (0,1): [('B00', 1), ('B01', 1)],
                (1,0): [('B00', 1), ('B01', 1)],
                (1,1): [('B00', 1), ('B01', 1)]
            }
            z_map = {
                (0,0): [('C11', 1)],
                (0,1): [('C11', 1)],
                (1,0): [('C11', 1)],
                (1,1): [('C11', 1)]
            }
            return x_map, y_map, z_map
        
        if term_idx == 6: # M7 = (A01-A11)(B10+B11)
            x_map = {
                (0,0): [('A01', 1), ('A11', -1)],
                (0,1): [('A01', 1), ('A11', -1)],
                (1,0): [('A01', 1), ('A11', -1)],
                (1,1): [('A01', 1), ('A11', -1)]
            }
            y_map = {
                (0,0): [('B10', 1), ('B11', 1)],
                (0,1): [('B10', 1), ('B11', 1)],
                (1,0): [('B10', 1), ('B11', 1)],
                (1,1): [('B10', 1), ('B11', 1)]
            }
            z_map = {
                (0,0): [('C00', 1)],
                (0,1): [('C00', 1)],
                (1,0): [('C00', 1)],
                (1,1): [('C00', 1)]
            }
            return x_map, y_map, z_map

    col = 0
    for t_idx in range(7):
        x_map, y_map, z_map = get_4x4_maps(t_idx)
        
        for s2_idx in range(7):
            x_coeffs, y_coeffs, z_contribs = s2_terms[s2_idx]
            
            # For this scalar multiplication, we need to add contributions to U, V, W
            # U corresponds to A
            # V corresponds to B
            # W corresponds to C
            
            # Iterate over all (r,c) in 2x2 for X
            for (xr, xc), x_coeff in x_coeffs.items():
                # Get the list of A_blocks contributing to X[xr,xc]
                for (a_block, a_coeff) in x_map[(xr, xc)]:
                    # a_block is like 'A00'
                    # Map to 4x4 indices
                    # A00 is rows 0-1, cols 0-1
                    # A01 is rows 0-1, cols 2-3
                    # A10 is rows 2-3, cols 0-1
                    # A11 is rows 2-3, cols 2-3
                    
                    # The element X[xr,xc] corresponds to A_block[xr,xc]
                    # So the 4x4 index is:
                    # Row: block_row * 2 + xr
                    # Col: block_col * 2 + xc
                    
                    # Parse block
                    b_r = int(a_block[1])
                    b_c = int(a_block[2])
                    
                    i = b_r * 2 + xr
                    j = b_c * 2 + xc
                    
                    # U[i * m + j] += x_coeff * a_coeff
                    U[i * m + j, col] += x_coeff * a_coeff
            
            # Iterate over all (r,c) in 2x2 for Y
            for (yr, yc), y_coeff in y_coeffs.items():
                for (b_block, b_coeff) in y_map[(yr, yc)]:
                    b_r = int(b_block[1])
                    b_c = int(b_block[2])
                    
                    # V index: row of B is yr, col is yc?
                    # In 2x2 Strassen, Y is the second matrix.
                    # Y[r,c] corresponds to B_block[r,c]
                    # Row of B: b_r * 2 + yr
                    # Col of B: b_c * 2 + yc
                    
                    i = b_r * 2 + yr
                    j = b_c * 2 + yc
                    
                    # V[i * p + j] += y_coeff * b_coeff
                    V[i * p + j, col] += y_coeff * b_coeff
            
            # Iterate over z_contribs
            for (zr, zc, z_coeff) in z_contribs:
                # z_contribs gives the contribution to Z[zr, zc]
                # We need to map this to C blocks
                for (c_block, c_coeff) in z_map[(zr, zc)]:
                    c_r = int(c_block[1])
                    c_c = int(c_block[2])
                    
                    # C block row: c_r, col: c_c
                    # Element Z[zr, zc] corresponds to C_block[zr, zc]
                    # Row of C: c_r * 2 + zr
                    # Col of C: c_c * 2 + zc
                    
                    i = c_r * 2 + zr
                    j = c_c * 2 + zc
                    
                    # W index: k * n + i ?
                    # Seed: W[k * n + i] where k is col of B (which is row of C in seed's logic? No)
                    # Seed: U[i,j] A, V[j,k] B, W[k,i] C?
                    # Let's re-verify seed's W index.
                    # Seed: W[k * n + i] = 1.
                    # k is from range(p) (col of B). i is from range(n) (row of A).
                    # So W index is (col_B, row_A).
                    # My C indices are (row_C, col_C).
                    # row_C corresponds to row_A (i).
                    # col_C corresponds to col_B (k).
                    # So W index should be (col_C, row_C) = (j, i).
                    # Index = j * n + i.
                    
                    W[j * n + i, col] += z_coeff * c_coeff
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

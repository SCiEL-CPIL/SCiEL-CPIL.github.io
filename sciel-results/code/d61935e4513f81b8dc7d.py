"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use recursive Strassen: 4x4 = 2x2 blocks of 2x2, each 2x2 uses Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Index helpers
    def u_idx(i, j): return i * m + j  # A[i,j]
    def v_idx(j, k): return j * p + k  # B[j,k]
    def w_idx(i, k): return k * n + i  # C[i,k] (trace convention: W[k*n+i])
    
    # 2x2 Strassen terms: for X=[[a,b],[c,d]], Y=[[e,f],[g,h]], Z=X*Y
    # t1 = (a+d)(e+h) -> z00 += 1, z11 += 1
    # t2 = (c+d)e     -> z10 += 1, z11 += -1
    # t3 = a(f-h)     -> z01 += 1, z11 += 1
    # t4 = d(g-e)     -> z00 += 1, z10 += 1
    # t5 = (a+b)h     -> z00 += -1, z01 += 1
    # t6 = (c-a)(e+f) -> z11 += 1
    # t7 = (b-d)(g+h) -> z00 += 1
    
    # Each inner Strassen term: (x_elements, x_coeffs, y_elements, y_coeffs, z_elements, z_coeffs)
    # x_elements: indices into [a,b,c,d] (0,1,2,3)
    # y_elements: indices into [e,f,g,h] (0,1,2,3)
    # z_elements: indices into [z00,z01,z10,z11] (0,1,2,3)
    
    inner_strassen = [
        # t1: (a+d)(e+h)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # t2: (c+d)e
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # t3: a(f-h)
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # t4: d(g-e)
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # t5: (a+b)h
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # t6: (c-a)(e+f)
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # t7: (b-d)(g+h)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # Block-level Strassen for 4x4:
    # M1 = (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
    # M2 = (A21+A22)*B11       -> C21 += 1, C22 += -1
    # M3 = A11*(B12-B22)       -> C12 += 1, C22 += 1
    # M4 = A22*(B21-B11)       -> C11 += 1, C21 += 1
    # M5 = (A11+A12)*B22       -> C11 += -1, C12 += 1
    # M6 = (A21-A11)*(B11+B12) -> C22 += 1
    # M7 = (A12-A22)*(B21+B22) -> C11 += 1
    
    # For each block term, define X (2x2 from A) and Y (2x2 from B)
    # X[r][c] and Y[r][c] are dicts: {element_index: coeff}
    # where element_index is u_idx(i,j) for X, v_idx(j,k) for Y
    
    def make_block_terms():
        terms = []
        
        # M1: X = A11+A22, Y = B11+B22
        X1 = [
            [{u_idx(0,0):1, u_idx(2,2):1}, {u_idx(0,1):1, u_idx(2,3):1}],
            [{u_idx(1,0):1, u_idx(3,2):1}, {u_idx(1,1):1, u_idx(3,3):1}]
        ]
        Y1 = [
            [{v_idx(0,0):1, v_idx(2,2):1}, {v_idx(0,1):1, v_idx(2,3):1}],
            [{v_idx(1,0):1, v_idx(3,2):1}, {v_idx(1,1):1, v_idx(3,3):1}]
        ]
        C1 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X1, Y1, C1))
        
        # M2: X = A21+A22, Y = B11
        X2 = [
            [{u_idx(2,0):1, u_idx(2,2):1}, {u_idx(2,1):1, u_idx(2,3):1}],
            [{u_idx(3,0):1, u_idx(3,2):1}, {u_idx(3,1):1, u_idx(3,3):1}]
        ]
        Y2 = [
            [{v_idx(0,0):1}, {v_idx(0,1):1}],
            [{v_idx(1,0):1}, {v_idx(1,1):1}]
        ]
        C2 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1, (2,2):-1, (2,3):-1, (3,2):-1, (3,3):-1}
        terms.append((X2, Y2, C2))
        
        # M3: X = A11, Y = B12-B22
        X3 = [
            [{u_idx(0,0):1}, {u_idx(0,1):1}],
            [{u_idx(1,0):1}, {u_idx(1,1):1}]
        ]
        Y3 = [
            [{v_idx(0,2):1, v_idx(2,2):-1}, {v_idx(0,3):1, v_idx(2,3):-1}],
            [{v_idx(1,2):1, v_idx(3,2):-1}, {v_idx(1,3):1, v_idx(3,3):-1}]
        ]
        C3 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1, (2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X3, Y3, C3))
        
        # M4: X = A22, Y = B21-B11
        X4 = [
            [{u_idx(2,2):1}, {u_idx(2,3):1}],
            [{u_idx(3,2):1}, {u_idx(3,3):1}]
        ]
        Y4 = [
            [{v_idx(2,0):1, v_idx(0,0):-1}, {v_idx(2,1):1, v_idx(0,1):-1}],
            [{v_idx(3,0):1, v_idx(1,0):-1}, {v_idx(3,1):1, v_idx(1,1):-1}]
        ]
        C4 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1, (2,0):1, (2,1):1, (3,0):1, (3,1):1}
        terms.append((X4, Y4, C4))
        
        # M5: X = A11+A12, Y = B22
        X5 = [
            [{u_idx(0,0):1, u_idx(0,2):1}, {u_idx(0,1):1, u_idx(0,3):1}],
            [{u_idx(1,0):1, u_idx(1,2):1}, {u_idx(1,1):1, u_idx(1,3):1}]
        ]
        Y5 = [
            [{v_idx(2,2):1}, {v_idx(2,3):1}],
            [{v_idx(3,2):1}, {v_idx(3,3):1}]
        ]
        C5 = {(0,0):-1, (0,1):-1, (1,0):-1, (1,1):-1, (0,2):1, (0,3):1, (1,2):1, (1,3):1}
        terms.append((X5, Y5, C5))
        
        # M6: X = A21-A11, Y = B11+B12
        X6 = [
            [{u_idx(2,0):1, u_idx(0,0):-1}, {u_idx(2,1):1, u_idx(0,1):-1}],
            [{u_idx(3,0):1, u_idx(1,0):-1}, {u_idx(3,1):1, u_idx(1,1):-1}]
        ]
        Y6 = [
            [{v_idx(0,0):1, v_idx(0,2):1}, {v_idx(0,1):1, v_idx(0,3):1}],
            [{v_idx(1,0):1, v_idx(1,2):1}, {v_idx(1,1):1, v_idx(1,3):1}]
        ]
        C6 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
        terms.append((X6, Y6, C6))
        
        # M7: X = A12-A22, Y = B21+B22
        X7 = [
            [{u_idx(0,2):1, u_idx(2,2):-1}, {u_idx(0,3):1, u_idx(2,3):-1}],
            [{u_idx(1,2):1, u_idx(3,2):-1}, {u_idx(1,3):1, u_idx(3,3):-1}]
        ]
        Y7 = [
            [{v_idx(2,0):1, v_idx(2,2):1}, {v_idx(2,1):1, v_idx(2,3):1}],
            [{v_idx(3,0):1, v_idx(3,2):1}, {v_idx(3,1):1, v_idx(3,3):1}]
        ]
        C7 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
        terms.append((X7, Y7, C7))
        
        return terms
    
    block_terms = make_block_terms()
    
    # For each block term, apply the 7 inner Strassen terms
    # Each inner term involves products of X elements and Y elements
    # We need to expand each product into individual rank-1 terms
    
    for block_idx, (X, Y, C_map) in enumerate(block_terms):
        base_col = block_idx * 7
        
        for s_idx, (x_elems, x_coeffs, y_elems, y_coeffs, z_elems, z_coeffs) in enumerate(inner_strassen):
            col = base_col + s_idx
            
            # For this inner Strassen term, we have:
            # x_sum = sum(x_coeffs[i] * X_element[x_elems[i]] for i in range(len(x_elems)))
            # y_sum = sum(y_coeffs[i] * Y_element[y_elems[i]] for i in range(len(y_elems)))
            # contribution = x_sum * y_sum
            # This contributes to Z elements z_elems[j] with coefficient z_coeffs[j]
            
            # X_element is a 2x2 matrix entry, which is a dict {u_idx: coeff}
            # Y_element is a 2x2 matrix entry, which is a dict {v_idx: coeff}
            
            # The product x_sum * y_sum is a bilinear form.
            # x_sum = sum over (xi, cx) of cx * A[xi]
            # y_sum = sum over (yi, cy) of cy * B[yi]
            # x_sum * y_sum = sum over (xi, yi) of (cx * cy) * A[xi] * B[yi]
            
            # In the tensor decomposition, each A[xi] * B[yi] term is a rank-1 term
            # But we can group them: the entire x_sum * y_sum is represented by
            # one column in U, V, W where U has the x coefficients, V has the y coefficients,
            # and W has the z coefficients.
            
            # Wait, that's not right. In the tensor decomposition, each rank-1 term is
            # U[row_u] * V[row_v] * W[row_w] for a single column. But U[row_u] can have
            # multiple non-zero entries (it's a vector). So one column can represent
            # a product of two linear combinations.
            
            # So for each inner Strassen term, we have ONE column (col) where:
            # U has entries for all A elements involved in x_sum
            # V has entries for all B elements involved in y_sum
            # W has entries for all C elements involved in the z contribution
            
            # Build U contributions for this column
            for i in range(len(x_elems)):
                x_elem_idx = x_elems[i]  # 0,1,2,3 -> position in 2x2 X
                x_coeff = x_coeffs[i]
                # X[x_elem_idx] is a dict {u_idx: coeff}
                # x_elem_idx: 0=X[0][0], 1=X[0][1], 2=X[1][0], 3=X[1][1]
                if x_elem_idx == 0:
                    elem_dict = X[0][0]
                elif x_elem_idx == 1:
                    elem_dict = X[0][1]
                elif x_elem_idx == 2:
                    elem_dict = X[1][0]
                elif x_elem_idx == 3:
                    elem_dict = X[1][1]
                
                for u_row, u_c in elem_dict.items():
                    U[u_row][col] += x_coeff * u_c
            
            # Build V contributions for this column
            for i in range(len(y_elems)):
                y_elem_idx = y_elems[i]
                y_coeff = y_coeffs[i]
                if y_elem_idx == 0:
                    elem_dict = Y[0][0]
                elif y_elem_idx == 1:
                    elem_dict = Y[0][1]
                elif y_elem_idx == 2:
                    elem_dict = Y[1][0]
                elif y_elem_idx == 3:
                    elem_dict = Y[1][1]
                
                for v_row, v_c in elem_dict.items():
                    V[v_row][col] += y_coeff * v_c
            
            # Build W contributions for this column
            # z_elems and z_coeffs tell us which Z elements get contributions
            # Z element indices: 0=z00, 1=z01, 2=z10, 3=z11
            # These correspond to block positions in the 2x2 result
            # We need to map these to actual C[i,k] positions using C_map
            
            for j in range(len(z_elems)):
                z_elem_idx = z_elems[j]
                z_coeff = z_coeffs[j]
                
                # Map z_elem_idx to block (br, bc)
                if z_elem_idx == 0:  # z00 -> block (0,0) -> C rows 0,1 cols 0,1
                    br, bc = 0, 0
                elif z_elem_idx == 1:  # z01 -> block (0,1) -> C rows 0,1 cols 2,3
                    br, bc = 0, 1
                elif z_elem_idx == 2:  # z10 -> block (1,0) -> C rows 2,3 cols 0,1
                    br, bc = 1, 0
                elif z_elem_idx == 3:  # z11 -> block (1,1) -> C rows 2,3 cols 2,3
                    br, bc = 1, 1
                
                # For each (i,k) in C_map, check if it falls in block (br, bc)
                for (ci, ck), c_coeff in C_map.items():
                    # Check if (ci, ck) is in block (br, bc)
                    if (ci // 2 == br) and (ck // 2 == bc):
                        w_row = w_idx(ci, ck)
                        W[w_row][col] += z_coeff * c_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

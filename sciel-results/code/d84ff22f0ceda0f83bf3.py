import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (24, rank)
    # W: (p*n, rank) -> (24, rank)
    
    # We will build lists of coefficients for U, V, W
    # Each term is a tuple (u_row, v_row, w_row, coeff_u, coeff_v, coeff_w)
    # Since all coeffs are -1, 0, 1, we can just store the rows and signs.
    
    terms = []
    
    # --- Part 1: Strassen for 4x4x4 (Left part of B) ---
    # A blocks:
    # A00: i in [0,1], j in [0,1]
    # A01: i in [0,1], j in [2,3]
    # A10: i in [2,3], j in [0,1]
    # A11: i in [2,3], j in [2,3]
    
    # B_left blocks:
    # B00: j in [0,1], k in [0,1]
    # B01: j in [0,1], k in [2,3]
    # B10: j in [2,3], k in [0,1]
    # B11: j in [2,3], k in [2,3]
    
    # Define the 7 Strassen combinations for the 4x4 level
    # Each combination is a list of (block_A, sign) and (block_B, sign)
    # block_A is one of 'A00', 'A01', 'A10', 'A11'
    # block_B is one of 'B00', 'B01', 'B10', 'B11'
    
    strassen_4x4_combos = [
        # M1 = (A00 + A11)(B00 + B11)
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)]),
        # M2 = (A10 + A11)B00
        ([('A10', 1), ('A11', 1)], [('B00', 1)]),
        # M3 = A00(B01 - B11)
        ([('A00', 1)], [('B01', 1), ('B11', -1)]),
        # M4 = A11(B10 - B00)
        ([('A11', 1)], [('B10', 1), ('B00', -1)]),
        # M5 = (A00 + A01)B11
        ([('A00', 1), ('A01', 1)], [('B11', 1)]),
        # M6 = (A10 - A00)(B00 + B01)
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)]),
        # M7 = (A01 - A11)(B10 + B11)
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)])
    ]
    
    # For each of these 7 "big" multiplications, we apply Strassen 2x2x2
    # The inputs to the 2x2x2 Strassen are the blocks themselves.
    # We need to map the elements of the blocks to the global indices.
    
    # Block index mapping:
    # A blocks:
    # A00: i=0,1; j=0,1
    # A01: i=0,1; j=2,3
    # A10: i=2,3; j=0,1
    # A11: i=2,3; j=2,3
    
    # B_left blocks:
    # B00: j=0,1; k=0,1
    # B01: j=0,1; k=2,3
    # B10: j=2,3; k=0,1
    # B11: j=2,3; k=2,3
    
    # We define a helper to get the (i, j) or (j, k) indices for a block element.
    # For A block 'Axy', element (r, c) maps to i = x*2 + r, j = y*2 + c.
    # For B block 'Bxy', element (r, c) maps to j = x*2 + r, k = y*2 + c.
    
    def get_a_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return x * 2 + r, y * 2 + c
        
    def get_b_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return x * 2 + r, y * 2 + c

    # Strassen 2x2x2 terms
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y01 - y11)
    # S4 = x11(y10 - y00)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    
    # Result Z = XY
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # We iterate over the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(strassen_4x4_combos):
        # For each big multiplication, we have 7 small Strassen terms
        # Each small term is a product of linear combinations of X elements and Y elements.
        # X elements come from the A block specified in a_parts.
        # Y elements come from the B block specified in b_parts.
        
        # We need to sum the contributions from all a_parts and b_parts.
        # Actually, the "big" multiplication is:
        # M = (sum a_sign * A_block) * (sum b_sign * B_block)
        # This expands to sum_{a,b} a_sign * b_sign * (A_block * B_block)
        # Each A_block * B_block is a 2x2x2 multiplication.
        # We can apply Strassen to each A_block * B_block pair?
        # No, Strassen is for the whole product.
        # Wait, Strassen's algorithm works on the blocks.
        # The 7 terms S1..S7 are computed from the elements of the blocks.
        # So for a specific big multiplication, say M1 = (A00+A11)(B00+B11),
        # The inputs to the 2x2 Strassen are the matrices X = A00+A11 and Y = B00+B11.
        # The elements of X are x_rc = (A00)_{rc} + (A11)_{rc}.
        # The elements of Y are y_rc = (B00)_{rc} + (B11)_{rc}.
        
        # So we can just define the coefficients for x_rc and y_rc for each small Strassen term.
        
        # Small Strassen term definitions:
        # Each term is a list of (x_index, coeff) and (y_index, coeff)
        # x_index is (r, c) in 0..1, y_index is (r, c) in 0..1
        
        small_strassen_terms = [
            # S1
            ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
            # S2
            ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
            # S3
            ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
            # S4
            ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
            # S5
            ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
            # S6
            ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
            # S7
            ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        ]
        
        # Result mapping:
        # Z00 = S1 + S4 - S5 + S7
        # Z01 = S3 + S5
        # Z10 = S2 + S4
        # Z11 = S1 - S2 + S3 + S6
        
        result_map = {
            (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)], # S1, S4, -S5, S7
            (0,1): [(2, 1), (4, 1)],                  # S3, S5
            (1,0): [(1, 1), (3, 1)],                  # S2, S4
            (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]  # S1, -S2, S3, S6
        }
        
        for r in range(2):
            for c in range(2):
                # This (r,c) is the position in the 2x2 result block Z.
                # We need to add the contributions from the small Strassen terms.
                for s_idx, coeff in result_map[(r,c)]:
                    x_coeffs, y_coeffs = small_strassen_terms[s_idx]
                    # Expand x_coeffs and y_coeffs to global A and B indices
                    # x_coeffs are ( (rx, cx), cx_coeff )
                    # We need to sum over the a_parts and b_parts.
                    
                    # For each x element, we have contributions from all a_parts.
                    # x_rc = sum_{a_block, a_sign} a_sign * A_block[rc]
                    # So for a specific x element (rx, cx) with coefficient cx_coeff,
                    # the total coefficient for A_block[rx, cx] is cx_coeff * a_sign.
                    
                    # Similarly for y.
                    
                    # This term is a product of:
                    # (sum_{a_block} a_sign * cx_coeff * A_block[rx, cx]) * (sum_{b_block} b_sign * cy_coeff * B_block[ry, cy])
                    # This is not a single scalar multiplication, it's a sum of products.
                    # Wait, Strassen's algorithm computes the product of two matrices.
                    # The "multiplication" is the scalar product of the linear combinations.
                    # So each small Strassen term S_s is ONE scalar multiplication.
                    # The scalar is the product of the two linear combinations.
                    
                    # So for each small Strassen term S_s, we have:
                    # U part: sum of (a_sign * cx_coeff * A_index)
                    # V part: sum of (b_sign * cy_coeff * B_index)
                    # W part: The result Z[rc] gets coefficient 1 from this S_s?
                    # No, the result Z[rc] is a sum of S_s.
                    # So for each S_s contributing to Z[rc], we add a term to the tensor.
                    
                    # The term is:
                    # u_row = A_index
                    # v_row = B_index
                    # w_row = C_index
                    # coeff_u = a_sign * cx_coeff
                    # coeff_v = b_sign * cy_coeff
                    # coeff_w = 1 (since we are adding S_s to Z[rc])
                    
                    # But wait, the "multiplication" is the product of the sums.
                    # (Sum U) * (Sum V) is not how tensor decomposition works.
                    # Tensor decomposition is sum of rank-1 tensors.
                    # Each rank-1 tensor is u * v * w.
                    # Here, the "multiplication" is the scalar product.
                    # So the term is:
                    # (sum_u u_i A_i) * (sum_v v_j B_j) * (w_k C_k)
                    # This is not a rank-1 tensor in the standard sense unless we fix the sums.
                    # Ah, Strassen's algorithm works by defining new variables.
                    # M1 = (A00+A11)(B00+B11).
                    # This M1 is a matrix.
                    # We compute M1 using 7 scalar multiplications.
                    # Each scalar multiplication is of the form (linear_comb_A) * (linear_comb_B).
                    # So each scalar multiplication corresponds to a set of rank-1 tensors?
                    # No.
                    # The product of two linear combinations is a single scalar.
                    # This scalar is then used to form the output matrix.
                    # So for each scalar multiplication, we have:
                    # u = vector of coefficients for A
                    # v = vector of coefficients for B
                    # w = vector of coefficients for C (which is the position in the output matrix)
                    
                    # So for each small Strassen term S_s:
                    # We form u_s and v_s.
                    # u_s is a vector of length 16 (for A).
                    # v_s is a vector of length 24 (for B).
                    # w_s is a vector of length 24 (for C).
                    
                    # The contribution to C[rc] is S_s * w_s[rc].
                    # But S_s is the product of the linear combinations.
                    # So the term in the tensor is u_s * v_s * w_s?
                    # No, the tensor is T = sum u_s v_s w_s.
                    # And C = (U * A) * (V * B) * W? No.
                    # The decomposition is:
                    # C_{ik} = sum_{s} (sum_{ij} U_{ij,s} A_{ij}) (sum_{jk} V_{jk,s} B_{jk}) W_{ik,s}
                    
                    # So for each s, we have:
                    # U_s is the vector of coefficients for A.
                    # V_s is the vector of coefficients for B.
                    # W_s is the vector of coefficients for C.
                    
                    # For Strassen, each scalar multiplication s defines U_s and V_s.
                    # And W_s is determined by how s contributes to the output blocks.
                    
                    # So for each small Strassen term s (0..6) in a big multiplication m:
                    # We compute U_s and V_s.
                    # Then we determine which C elements receive this term.
                    # The contribution to C[rc] is coeff * S_s.
                    # So W_s[rc] = coeff.
                    
                    # Let's refine this.
                    pass

    # This logic is getting complex to implement correctly in a short time.
    # The rank 81 solution is solid.
    # I will implement the rank 81 solution.
    
    # Initialize U, V, W with zeros
    rank = 81
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    
    # --- Part 1: Strassen 4x4x4 ---
    # We will generate the 49 terms.
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            U[i][col] += c
        for j, c in v_coeffs:
            V[j][col] += c
        for k, c in w_coeffs:
            W[k][col] += c
        col += 1

    # Define block index mapping again
    # A block 'Axy' element (r,c) -> i = x*2+r, j = y*2+c -> index i*4+j
    # B block 'Bxy' element (r,c) -> j = x*2+r, k = y*2+c -> index j*6+k
    
    def a_block_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return (x*2+r)*4 + (y*2+c)
        
    def b_block_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return (x*2+r)*6 + (y*2+c)
        
    def c_block_idx(block, r, c):
        # C is 4x4 for the left part
        # C block 'Cxy' element (r,c) -> i = x*2+r, k = y*2+c
        # W index is k*4+i
        x, y = int(block[1]), int(block[2])
        return (y*2+c)*4 + (x*2+r)

    # Strassen 4x4 combinations
    strassen_4x4_combos = [
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)]),
        ([('A10', 1), ('A11', 1)], [('B00', 1)]),
        ([('A00', 1)], [('B01', 1), ('B11', -1)]),
        ([('A11', 1)], [('B10', 1), ('B00', -1)]),
        ([('A00', 1), ('A01', 1)], [('B11', 1)]),
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)]),
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)])
    ]
    
    # Small Strassen terms
    # Each term is (x_coeffs, y_coeffs)
    # x_coeffs: list of ( (r,c), coeff )
    # y_coeffs: list of ( (r,c), coeff )
    
    small_strassen_terms = [
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    # Result mapping for 2x2 Strassen
    # Z[rc] = sum coeff * S_s
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(strassen_4x4_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = small_strassen_terms[s_idx]
            
            # Construct U_coeffs for this term
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_block_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            # Construct V_coeffs for this term
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b_block_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            # Construct W_coeffs for this term
            # This term S_s contributes to Z[rc] with certain coefficients.
            # We need to add a term to the tensor for each rc where coeff != 0.
            # But wait, the tensor decomposition sums over s.
            # So for each s, we have one term in the sum.
            # The W vector for this term s has non-zero entries at the positions rc.
            
            w_coeffs = []
            for rc, coeff in result_map.items():
                # Find the coefficient of S_s in Z[rc]
                # result_map[rc] is a list of (s_idx, coeff)
                # We need to check if s_idx is in this list.
                pass # This logic is wrong. result_map maps rc to list of (s_idx, coeff).
                # We need to invert it or check for each rc.
            
            # Let's restructure.
            # For each rc, we know which s_idx contribute.
            # So for a fixed s_idx, we can check which rc it contributes to.
            
            for rc in result_map:
                for s_check, c_val in result_map[rc]:
                    if s_check == s_idx:
                        # This s_idx contributes to rc with coefficient c_val.
                        # The C block is C_big_idx? No.
                        # The big multiplication M_big_idx produces a 2x2 block of C.
                        # Which block?
                        # M1 -> C00
                        # M2 -> C10
                        # M3 -> C01
                        # M4 -> C11? No.
                        # Let's check the Strassen 4x4 result formula.
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # So M_big_idx contributes to C_block with certain coefficients.
                        pass

    # This is getting too complex to debug in my head.
    # I will use a simpler, verified rank 81 construction.
    # I'll just output the schoolbook rank 96 if I can't get Strassen right quickly.
    # But 81 is a good target.
    
    # Let's try a different approach.
    # Just return the schoolbook solution.
    # It's safe and correct.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

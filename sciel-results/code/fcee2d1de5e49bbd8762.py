import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen 2x2 blocks)
    # Rank 16 for 4x4x1 part (Schoolbook)
    R_444 = 49
    R_441 = 16
    rank = R_444 + R_441
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index in U (A is n x m)
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get index in V (B is m x p)
    def v_idx(j, k):
        return j * p + k
        
    # Helper to get index in W (C is p x n, index k*n+i)
    def w_idx(k, i):
        return k * n + i

    # --- Part 1: 4x4x4 using Strassen on 2x2 blocks ---
    # Strassen coefficients for 2x2 matrix multiplication
    # S0..S6
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    # We define the 7 Strassen products for a generic 2x2 multiplication
    # Each product s has coefficients for X (U) and Y (V)
    # And the result Z is linear combination of S0..S6
    
    # Strassen Formulas:
    # S0 = (x00+x11)(y00+y11)
    # S1 = (x10+x11)y00
    # S2 = x00(y01-y11)
    # S3 = x11(y10-y00)
    # S4 = (x00+x01)y11
    # S5 = (x10-x00)(y00+y01)
    # S6 = (x01-x11)(y10+y11)
    
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5
    
    # We will iterate over the 7 block multiplications M0..M6
    # For each Mk, we iterate over the 7 Strassen steps s=0..6
    # Total 49 terms.
    
    # Define the block combinations for M0..M6
    # M0 = (A00 + A11) (B00 + B11)
    # M1 = (A10 + A11) B00
    # M2 = A00 (B01 - B11)
    # M3 = A11 (B10 - B00)
    # M4 = (A00 + A01) B11
    # M5 = (A10 - A00) (B00 + B01)
    # M6 = (A01 - A11) (B10 + B11)
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # Note: B is 4x5. For the 4x4 part, we use cols 0-3.
    
    # Block indices (I, J)
    # A_IJ is 2x2.
    # A_IJ[i, j] is at global A[2I+i, 2J+j]
    
    # For each block multiplication k (0..6):
    # X is sum of alpha_A[I,J] * A_IJ
    # Y is sum of alpha_B[I,J] * B_IJ
    
    block_defs = [
        # (coeffs for A blocks, coeffs for B blocks)
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),            # M2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),            # M3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),  # M5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})   # M6
    ]
    
    # Strassen step definitions for X, Y
    # Each step s defines:
    # U_coeffs: { (i,j): coeff } for X
    # V_coeffs: { (i,j): coeff } for Y
    # W_coeffs: { (i,j): coeff } for Z (result of this step)
    # Note: Z is 2x2.
    
    strassen_steps = [
        # S0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # S1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}), # Wait, Z10 and Z11 use S1?
        # Let's re-verify Z formulas.
        # Z00 = S0 + S3 - S4 + S6
        # Z01 = S2 + S4
        # Z10 = S1 + S3
        # Z11 = S0 - S1 + S2 + S5
        # So S1 contributes to Z10 and Z11.
        # S1 = (x10+x11)y00
        # So U: x10, x11. V: y00.
        # W: Z10, Z11.
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),
        # S2 = x00(y01-y11)
        # W: Z01, Z11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # S3 = x11(y10-y00)
        # W: Z00, Z10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # S4 = (x00+x01)y11
        # W: Z00, Z01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
        # S5 = (x10-x00)(y00+y01)
        # W: Z11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # S6 = (x01-x11)(y10+y11)
        # W: Z00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col_offset = 0
    for k in range(7):
        a_coeffs, b_coeffs = block_defs[k]
        for s in range(7):
            u_local, v_local, w_local = strassen_steps[s]
            
            # Fill U
            # U maps global A to this rank term
            # X_ij = sum_{I,J} a_coeffs[(I,J)] * A_{2I+i, 2J+j}
            for (I, J), coeff_a in a_coeffs.items():
                for (i, j), coeff_u in u_local.items():
                    global_i = 2 * I + i
                    global_j = 2 * J + j
                    U[u_idx(global_i, global_j)][col_offset] += coeff_a * coeff_u
            
            # Fill V
            # Y_ij = sum_{I,J} b_coeffs[(I,J)] * B_{2I+i, 2J+j}
            # Note: B is 4x5. We are using cols 0-3.
            for (I, J), coeff_b in b_coeffs.items():
                for (i, j), coeff_v in v_local.items():
                    global_j = 2 * I + i # Row index in B is 2I+i
                    global_k = 2 * J + j # Col index in B is 2J+j
                    V[v_idx(global_j, global_k)][col_offset] += coeff_b * coeff_v
            
            # Fill W
            # Z_ij contributes to C.
            # C_{2I+i, 2J+j} += w_local[(i,j)] * (product)
            # Wait, the result of block multiplication Mk is a 2x2 matrix Zk.
            # Zk is added to C blocks.
            # C = sum Mk.
            # So C_{2I+i, 2J+j} gets contribution from Zk_{i,j} if the block Mk corresponds to C block (I,J).
            # In Strassen, Mk is the result of the block multiplication.
            # The block index of the result is the same as the input blocks?
            # No, Strassen computes the whole C.
            # M0..M6 are intermediate 2x2 matrices.
            # C00 = M0[0,0] + M1[1,0] - M2[0,1] + M3[0,0]? No.
            # C is reconstructed from M0..M6.
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0]?
            # Let's use the standard Strassen output formulas for the 2x2 block case.
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0] -- This is wrong.
            
            # Standard Strassen for 2x2 blocks:
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0] is not right.
            # C00 = M0[0,0] + M3[1,1] ... no.
            
            # Let's re-derive.
            # C = [[C00, C01], [C10, C11]]
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0] ?
            # No, the formulas are:
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0] is for the 2x2 case where M are scalars.
            # Here M are 2x2 matrices.
            # The formulas apply element-wise.
            # C00[0,0] = M0[0,0][0,0] + M3[1,1][0,0] - M4[1,1][0,0] + M6[0,0][0,0]
            
            # So, for a specific Strassen step s in block k:
            # The product P = X*Y is a 2x2 matrix.
            # P is one of the M_k.
            # The contribution of P to C is determined by the Strassen output formulas.
            # But we are decomposing P itself into rank-1 tensors.
            # So for this rank term, we have a scalar product val.
            # val contributes to P[i,j] with weight w_local[(i,j)].
            # So P[i,j] += w_local[(i,j)] * val.
            # Then C gets P according to the block Strassen formulas.
            
            # We need to know which C block and which element of that block P[i,j] contributes to.
            # This depends on k (which M) and i,j (which element of M).
            
            # Let's pre-define the mapping from (k, i, j) to (C_block_I, C_block_J, C_elem_i, C_elem_j, coeff)
            
            # Strassen Output Formulas for 2x2 blocks:
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0]
            # C01 = M2[0,1] + M4[1,1]  -- Wait, M4[1,1] is used in C00 too?
            # Let's check standard Strassen:
            # C11 = M1[0,0] + M3[1,1]
            # C12 = M2[0,1] + M4[1,1]
            # C21 = M1[1,0] + M3[1,1] -- No, indices are 1-based in some texts.
            
            # Using 0-based:
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0]
            # C01 = M2[0,1] + M4[1,1]
            # C10 = M1[1,0] + M3[1,1]
            # C11 = M0[0,0] - M1[1,0] + M2[0,1] + M5[1,1]
            
            # Wait, M1 is (A21+A22)B11.
            # M1[1,0] is the bottom-left element of M1.
            
            # Let's verify C00.
            # A00B00 + A01B10 + A10B01 + A11B11.
            # M0 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # M3 = A11(B10-B00) = A11B10 - A11B00
            # M4 = (A00+A01)B11 = A00B11 + A01B11
            # M6 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
            
            # C00 = M0[0,0] + M3[1,1] - M4[1,1] + M6[0,0]
            # M0[0,0] = A00B00 + A00B11 + A11B00 + A11B11
            # M3[1,1] = A11B10 - A11B00
            # M4[1,1] = A00B11 + A01B11
            # M6[0,0] = A01B10 + A01B11 - A11B10 - A11B11
            # Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
            # = A00B00 + A01B10 + A10B01 + A11B11. Correct.
            
            # So the mapping is:
            # M0[0,0] -> C00[0,0] coeff 1
            # M3[1,1] -> C00[0,0] coeff 1
            # M4[1,1] -> C00[0,0] coeff -1
            # M6[0,0] -> C00[0,0] coeff 1
            
            # M2[0,1] -> C01[0,0] coeff 1
            # M4[1,1] -> C01[0,0] coeff 1
            
            # M1[1,0] -> C10[0,0] coeff 1
            # M3[1,1] -> C10[0,0] coeff 1
            
            # M0[0,0] -> C11[0,0] coeff 1
            # M1[1,0] -> C11[0,0] coeff -1
            # M2[0,1] -> C11[0,0] coeff 1
            # M5[1,1] -> C11[0,0] coeff 1
            
            # Note: The element indices (0,0) etc refer to the 2x2 submatrix.
            # So M_k[i,j] contributes to C_{I,J}[i,j] where I,J is the block index of C.
            # In the formulas above, C00 is the top-left block.
            # So M0[0,0] contributes to C[0,0] (top-left element of top-left block).
            
            # We need a table:
            # (k, i, j) -> (C_block_I, C_block_J, C_elem_i, C_elem_j, coeff)
            
            w_map = {
                (0, 0, 0): [(0, 0, 0, 0, 1)], # M0[0,0] -> C00[0,0]
                (3, 1, 1): [(0, 0, 0, 0, 1)], # M3[1,1] -> C00[0,0]
                (4, 1, 1): [(0, 0, 0, 0, -1)], # M4[1,1] -> C00[0,0]
                (6, 0, 0): [(0, 0, 0, 0, 1)], # M6[0,0] -> C00[0,0]
                
                (2, 0, 1): [(0, 1, 0, 0, 1)], # M2[0,1] -> C01[0,0]
                (4, 1, 1): [(0, 1, 0, 0, 1)], # M4[1,1] -> C01[0,0]
                
                (1, 1, 0): [(1, 0, 0, 0, 1)], # M1[1,0] -> C10[0,0]
                (3, 1, 1): [(1, 0, 0, 0, 1)], # M3[1,1] -> C10[0,0]
                
                (0, 0, 0): [(1, 1, 0, 0, 1)], # M0[0,0] -> C11[0,0]
                (1, 1, 0): [(1, 1, 0, 0, -1)], # M1[1,0] -> C11[0,0]
                (2, 0, 1): [(1, 1, 0, 0, 1)], # M2[0,1] -> C11[0,0]
                (5, 1, 1): [(1, 1, 0, 0, 1)]  # M5[1,1] -> C11[0,0]
            }
            
            # Wait, the mapping is not 1-to-1. A key in w_map can have multiple targets.
            # And a target can have multiple sources.
            # My w_map definition above is (k,i,j) -> list of (CI, CJ, ci, cj, coeff).
            
            # For this specific rank term (k, s):
            # We have w_local[(i,j)] for the 2x2 result of this Strassen step.
            # So for each (i,j) in w_local:
            #   coeff_w = w_local[(i,j)]
            #   Look up w_map.get((k, i, j), [])
            #   For each (CI, CJ, ci, cj, coeff_c) in that list:
            #       global_ci = 2*CI + ci
            #       global_cj = 2*CJ + cj
            #       W[w_idx(global_cj, global_ci)][col_offset] += coeff_w * coeff_c
            
            for (i, j), coeff_w in w_local.items():
                targets = w_map.get((k, i, j), [])
                for (CI, CJ, ci, cj, coeff_c) in targets:
                    global_ci = 2 * CI + ci
                    global_cj = 2 * CJ + cj
                    W[w_idx(global_cj, global_ci)][col_offset] += coeff_w * coeff_c
            
            col_offset += 1
            
    # --- Part 2: 4x4x1 (Schoolbook) ---
    # C[:, 4] = A @ B[:, 4]
    # C_i4 = sum_j A_ij B_j4
    # Rank 16.
    for i in range(4):
        for j in range(4):
            r = col_offset + 4 * i + j
            U[u_idx(i, j)][r] = 1
            V[v_idx(j, 4)][r] = 1
            W[w_idx(4, i)][r] = 1
            
    return {"U": U, "V": V, "W": W}

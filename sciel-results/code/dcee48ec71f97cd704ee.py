"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=4, m=4, p=6
    # U size: 16 x rank
    # V size: 24 x rank
    # W size: 24 x rank
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: length 16 (n*m)
        # v_vec: length 24 (m*p)
        # w_vec: length 24 (p*n)
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # --- Part 1: 4x4x4 decomposition (Rank 49) ---
    # Indices for 4x4 matrix:
    # A: i,j in 0..3. Index = i*4 + j
    # B: j,k in 0..3. Index = j*4 + k (k is 0..3)
    # C: k,i in 0..3. Index = k*4 + i
    
    # Strassen 2x2x2 terms (local indices 0..3)
    # Returns list of (u_local, v_local, w_local)
    def get_strassen_2x2():
        terms = []
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]))
        # M5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]))
        # M6
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 1, 0]))
        return terms

    strassen_terms = get_strassen_2x2()
    
    # Block level Strassen for 4x4
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # Define the 7 block-level products
    # Each product is (A_combo, B_combo, C_combo)
    # A_combo: list of (block_idx, coeff)
    # B_combo: list of (block_idx, coeff)
    # C_combo: list of (block_idx, coeff)
    
    block_products = [
        # M1: (A00+A11)(B00+B11) -> C00 + C11 (with signs in expansion)
        # Actually, we need to map the 2x2 Strassen output to the 4x4 C blocks.
        # The 2x2 Strassen computes C_block = X_block * Y_block.
        # X_block is a linear combo of A blocks.
        # Y_block is a linear combo of B blocks.
        # The result C_block is added to the global C with specific weights.
        
        # Let's define the 7 steps of 4x4 Strassen:
        # 1. X = A00 + A11, Y = B00 + B11. Result C_temp.
        #    C00 += C_temp_00, C11 += C_temp_11, C01 += C_temp_01? No.
        #    The 2x2 Strassen computes the full 2x2 product.
        #    So C_temp is a 2x2 matrix.
        #    C00 (global) gets C_temp_00.
        #    C11 (global) gets C_temp_11.
        #    Wait, in 4x4 Strassen:
        #    C00 = M1_00 - M4_00 + M5_00 - M7_00
        #    C01 = M3_00 + M4_00  (Wait, M3 is 2x2 product, so M3_00 is top-left of M3)
        #    Actually, the formula C00 = M1 - M4 + M5 - M7 refers to the 2x2 matrices.
        #    So C00 (4x4 block) = M1_00 - M4_00 + M5_00 - M7_00.
        #    C01 (4x4 block) = M3_00 + M4_00.
        #    C10 (4x4 block) = M2_00 + M7_00.
        #    C11 (4x4 block) = M1_11 - M2_11 + M3_11 + M6_11.
        
        # So for each of the 7 M_k (which are 2x2 products), we need to know
        # which part of the 4x4 C it contributes to, and with what sign.
        
        # M1: X = A00+A11, Y = B00+B11.
        #      Contributes to C00 (idx 0 of 2x2 result) with +1.
        #      Contributes to C11 (idx 3 of 2x2 result) with +1.
        #      Wait, C00 is the top-left 2x2 block of C.
        #      The 2x2 result of M1 is a 2x2 matrix.
        #      Its top-left element goes to C00_00?
        #      No, C00 is a 2x2 block.
        #      The formula C00 = M1_00 ... means the top-left 2x2 block of M1?
        #      No, M1 is a 2x2 matrix. M1_00 is its top-left element.
        #      But C00 is a 2x2 matrix.
        #      This implies the Strassen algorithm for 4x4 uses 2x2 matrix multiplications.
        #      The result of a 2x2 matrix multiplication is a 2x2 matrix.
        #      So M1 is a 2x2 matrix.
        #      C00 (2x2 block) = M1 (2x2 matrix) - M4 (2x2 matrix) + ...
        #      Yes, the linear combination is of the 2x2 matrices.
        
        # So, for each of the 7 M_k:
        # 1. Determine X (linear combo of 4 A blocks) and Y (linear combo of 4 B blocks).
        # 2. Decompose the 2x2 product X*Y into 7 scalar terms using strassen_terms.
        # 3. For each scalar term, determine its contribution to the 4 C blocks.
        
        # M1: X = A00+A11, Y = B00+B11.
        #      X_blocks: [(0, 1), (3, 1)]
        #      Y_blocks: [(0, 1), (3, 1)]
        #      C_blocks: C00 gets +1 * M1, C11 gets +1 * M1.
        #      So for each scalar term in M1, add to C00 and C11.
        
        # M2: X = A10+A11, Y = B00.
        #      X_blocks: [(2, 1), (3, 1)]
        #      Y_blocks: [(0, 1)]
        #      C_blocks: C10 gets +1 * M2, C11 gets -1 * M2.
        
        # M3: X = A00, Y = B01-B11.
        #      X_blocks: [(0, 1)]
        #      Y_blocks: [(1, 1), (3, -1)]
        #      C_blocks: C01 gets +1 * M3, C11 gets +1 * M3.
        
        # M4: X = A11, Y = B10-B00.
        #      X_blocks: [(3, 1)]
        #      Y_blocks: [(2, 1), (0, -1)]
        #      C_blocks: C00 gets -1 * M4, C01 gets +1 * M4.
        
        # M5: X = A00+A01, Y = B11.
        #      X_blocks: [(0, 1), (1, 1)]
        #      Y_blocks: [(3, 1)]
        #      C_blocks: C00 gets +1 * M5.
        
        # M6: X = A10-A00, Y = B00+B01.
        #      X_blocks: [(2, 1), (0, -1)]
        #      Y_blocks: [(0, 1), (1, 1)]
        #      C_blocks: C11 gets +1 * M6.
        
        # M7: X = A01-A11, Y = B10+B11.
        #      X_blocks: [(1, 1), (3, -1)]
        #      Y_blocks: [(2, 1), (3, 1)]
        #      C_blocks: C00 gets -1 * M7, C10 gets +1 * M7.
    ]
    
    # We need to map the 2x2 Strassen scalar terms to the 4x4 global indices.
    # The 2x2 Strassen terms produce a 2x2 result matrix.
    # The scalar terms correspond to the 4 entries of the 2x2 result.
    # w_local in strassen_terms is length 4: [C00, C01, C10, C11] of the 2x2 result.
    
    # For each of the 7 block products:
    for idx, (x_blocks, y_blocks, c_map) in enumerate([
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], {0: 1, 3: 1}), # M1
        ([(2, 1), (3, 1)], [(0, 1)], {2: 1, 3: -1}), # M2
        ([(0, 1)], [(1, 1), (3, -1)], {1: 1, 3: 1}), # M3
        ([(3, 1)], [(2, 1), (0, -1)], {0: -1, 1: 1}), # M4
        ([(0, 1), (1, 1)], [(3, 1)], {0: 1}), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], {3: 1}), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], {0: -1, 2: 1}) # M7
    ]):
        if time.time() > deadline: break
        
        # Build U and V vectors for the 2x2 block inputs
        # U_local length 4, V_local length 4
        # But we need to expand to 4x4 global indices.
        
        # For each scalar term in the 2x2 Strassen decomposition:
        for u_loc, v_loc, w_loc in strassen_terms:
            # u_loc is length 4 (A00, A01, A10, A11)
            # v_loc is length 4 (B00, B01, B10, B11)
            # w_loc is length 4 (C00, C01, C10, C11)
            
            # Expand u_loc to global U (length 16)
            u_global = [0] * 16
            for b_idx, coeff in x_blocks:
                # b_idx is 0,1,2,3 corresponding to A00, A01, A10, A11
                # The 2x2 Strassen term u_loc has entries for A00, A01, A10, A11
                # So we multiply coeff (from block) by u_loc[b_idx]
                # And place in the correct 4 slots
                base = b_idx * 4
                for k in range(4):
                    u_global[base + k] += coeff * u_loc[k]
                    
            # Expand v_loc to global V (length 24, but only first 16 used for 4x4x4)
            v_global = [0] * 24
            for b_idx, coeff in y_blocks:
                base = b_idx * 4
                for k in range(4):
                    v_global[base + k] += coeff * v_loc[k]
                    
            # Expand w_loc to global W (length 24)
            w_global = [0] * 24
            # w_loc[0] is C00 of the 2x2 result.
            # c_map tells which 4x4 C block this 2x2 result contributes to.
            # c_map keys are 0,1,2,3 corresponding to C00, C01, C10, C11.
            for c_idx, coeff in c_map.items():
                # c_idx is 0,1,2,3
                # The 2x2 result has 4 entries: 0,1,2,3
                # w_loc[j] is the coefficient for the j-th entry of the 2x2 result.
                # So for each j in 0..3:
                #   The entry j of the 2x2 result goes to the j-th slot of the c_idx-th 4x4 block.
                base = c_idx * 4
                for j in range(4):
                    w_global[base + j] += coeff * w_loc[j]
                    
            add_term(u_global, v_global, w_global)

    # --- Part 2: 4x4x2 decomposition (Rank 32) ---
    # Schoolbook for cols 4,5 of B
    # i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                if time.time() > deadline: break
                u_global = [0] * 16
                v_global = [0] * 24
                w_global = [0] * 24
                
                u_global[i * 4 + j] = 1
                v_global[j * 6 + k] = 1
                w_global[k * 4 + i] = 1
                
                add_term(u_global, v_global, w_global)

    rank = len(U_list)
    U = [list(map(float, row)) for row in U_list]
    V = [list(map(float, row)) for row in V_list]
    W = [list(map(float, row)) for row in W_list]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

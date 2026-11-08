import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank_total = 57  # 49 for 4x4x4, 8 for 4x4x2
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Outer Strassen terms for 2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # Indices: 
    # A row i, col j -> i*m + j
    # B row i, col j -> i*p + j
    # C row i, col j -> j*n + i (Trace convention)
    
    # Helper to add a linear combination to a matrix column
    def add_combo(mat, coeffs, c_idx, val=1):
        # coeffs: list of (local_index, coeff_val)
        # This function is for local 2x2 blocks in the intermediate step?
        # Actually, we need to map 2 levels of decomposition.
        pass

    # We will generate the 49 terms explicitly.
    # Outer terms (k=1..7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A01)(B11+B01)
    
    # Define outer linear combos for U (A) and V (B)
    # Format: list of (block_row, block_col, coeff)
    # Block (br, bc) corresponds to A[2*br:2*br+2, 2*bc:2*bc+2]
    
    outer_U = [
        # M1
        [(0,0,1), (1,1,1)],
        # M2
        [(1,0,1), (1,1,1)],
        # M3
        [(0,0,1)],
        # M4
        [(1,1,1)],
        # M5
        [(0,0,1), (1,0,1)],
        # M6
        [(0,1,1), (0,0,-1)],
        # M7
        [(1,1,1), (0,1,-1)]
    ]
    
    outer_V = [
        # M1
        [(0,0,1), (1,1,1)],
        # M2
        [(0,0,1)],
        # M3
        [(1,0,1), (1,1,-1)],
        # M4
        [(0,1,1), (0,0,-1)],
        # M5
        [(1,1,1)],
        # M6
        [(0,0,1), (1,0,1)],
        # M7
        [(1,1,1), (0,1,1)]
    ]
    
    # Outer W coeffs: how M_k contributes to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Format: list of (block_row, block_col, coeff)
    outer_W = [
        # M1
        [(0,0,1), (1,1,1)],
        # M2
        [(1,0,1), (1,1,-1)],
        # M3
        [(0,1,1), (1,1,1)],
        # M4
        [(0,0,1), (1,0,1)],
        # M5
        [(0,0,-1), (0,1,1)],
        # M6
        [(1,1,1)],
        # M7
        [(0,0,1), (1,1,1)]
    ]

    # Inner Strassen terms for 2x2x2 (Rank 7)
    # For a product X*Y (2x2 * 2x2):
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y10-y11)
    # m4 = x11(y01-y00)
    # m5 = (x00+x10)y11
    # m6 = (x01-x00)(y00+y10)
    # m7 = (x11-x01)(y11+y01)
    
    # Coeffs for X (U-part) and Y (V-part) in local 2x2 indices
    # Local index: row*2 + col
    inner_U = [
        [(0,1), (3,1)],  # m1: x00, x11
        [(2,1), (3,1)],  # m2: x10, x11
        [(0,1)],         # m3: x00
        [(3,1)],         # m4: x11
        [(0,1), (2,1)],  # m5: x00, x10
        [(1,1), (0,-1)], # m6: x01, -x00
        [(3,1), (1,-1)]  # m7: x11, -x01
    ]
    
    inner_V = [
        [(0,1), (3,1)],  # m1: y00, y11
        [(0,1)],         # m2: y00
        [(2,1), (3,-1)], # m3: y10, -y11
        [(1,1), (0,-1)], # m4: y01, -y00
        [(3,1)],         # m5: y11
        [(0,1), (2,1)],  # m6: y00, y10
        [(3,1), (1,1)]   # m7: y11, y01
    ]
    
    inner_W = [
        [(0,1), (3,1)],  # m1 -> c00, c11
        [(2,1), (3,-1)], # m2 -> c10, -c11
        [(1,1), (3,1)],  # m3 -> c01, c11
        [(0,1), (2,1)],  # m4 -> c00, c10
        [(0,-1), (1,1)], # m5 -> -c00, c01
        [(3,1)],         # m6 -> c11
        [(0,1), (3,1)]   # m7 -> c00, c11
    ]

    for k in range(7):
        u_outer = outer_U[k]
        v_outer = outer_V[k]
        w_outer = outer_W[k]
        
        for l in range(7):
            u_inner = inner_U[l]
            v_inner = inner_V[l]
            w_inner = inner_W[l]
            
            # Process U
            # u_outer defines which 2x2 blocks of A are involved and with what sign.
            # u_inner defines which elements within those blocks are involved.
            # We must combine them.
            # This is a bit tricky because u_outer sums blocks, and u_inner sums elements.
            # The product is sum_{blocks} sum_{elements} coeff_block * coeff_elem * A_elem
            
            # We can just iterate all combinations?
            # No, u_outer is a list of (br, bc, coeff).
            # u_inner is a list of (local_idx, coeff).
            # The total coefficient for a global A element is the sum over all pairs (outer, inner) that map to it.
            
            # Actually, the decomposition is:
            # Term = (Sum_outer U_block) * (Sum_outer V_block) ... 
            # Wait, the Strassen formula is linear.
            # M_k = U_k V_k.
            # U_k is a 2x2 matrix of A-blocks.
            # V_k is a 2x2 matrix of B-blocks.
            # Then M_k = Sum_l m_{k,l} where m_{k,l} are scalar products.
            # m_{k,l} = (Sum_{(i,j) in U_inner} coeff_{i,j} U_{i,j}) * (Sum_{(i,j) in V_inner} coeff'_{i,j} V_{i,j})
            
            # So, for the global A matrix:
            # We need to add contributions to U[col]
            
            # For each (br, bc, c_o) in u_outer:
            #   The block A[2*br:2*br+2, 2*bc:2*bc+2] is scaled by c_o.
            #   Inside this block, for each (li, c_i) in u_inner:
            #      The element at local index li in that block is scaled by c_i.
            #      Global index: row = 2*br + li//2, col = 2*bc + li%2
            #      Global U index: row*m + col
            
            for br, bc, c_o in u_outer:
                for li, c_i in u_inner:
                    r = 2 * br + li // 2
                    c = 2 * bc + li % 2
                    idx = r * m + c
                    U[idx][col] += c_o * c_i
            
            # Similarly for V
            for br, bc, c_o in v_outer:
                for li, c_i in v_inner:
                    r = 2 * br + li // 2
                    c = 2 * bc + li % 2
                    # V is 4x6. But here we are in the 4x4 part, so cols 0-3.
                    idx = r * p + c
                    V[idx][col] += c_o * c_i
            
            # Similarly for W
            # w_outer gives the target C-blocks.
            # w_inner gives the local elements within the result block M_k.
            # M_k is a 2x2 matrix.
            # The result of the scalar product m_{k,l} is added to specific elements of M_k.
            # Then M_k is added to specific C-blocks with specific signs.
            
            # For each (br, bc, c_o) in w_outer:
            #   This specifies that the result M_k is added to C-block (br, bc) with factor c_o.
            #   Inside M_k, for each (li, c_i) in w_inner:
            #      The local element li of M_k gets the scalar product.
            #      Global C index: row = 2*br + li//2, col = 2*bc + li%2
            #      W index (trace): col*n + row
            
            for br, bc, c_o in w_outer:
                for li, c_i in w_inner:
                    r = 2 * br + li // 2
                    c = 2 * bc + li % 2
                    idx = c * n + r
                    W[idx][col] += c_o * c_i
            
            col += 1
            
    # Part 2: 4x4x2 decomposition (Rank 8)
    # Columns 4 and 5 of B (0-indexed).
    # For each col_offset in [4, 5]:
    #   For j in 0..3:
    #     Term: A[:, j] * B[j, col_offset] -> C[:, col_offset]
    
    for col_offset in [4, 5]:
        for j in range(4):
            # U: extract A[:, j]
            for i in range(4):
                U[i * m + j][col] = 1
            # V: extract B[j, col_offset]
            V[j * p + col_offset][col] = 1
            # W: add to C[:, col_offset]
            for i in range(4):
                W[col_offset * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

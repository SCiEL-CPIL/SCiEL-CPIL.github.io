import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: M_4,4,4 using Strassen (Rank 49)
    # We need to construct U, V, W for the 4x4x4 part
    # Then add the 4x4x3 part (Rank 48)
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Construct M_4,4,4 part ---
    # Strassen on 2x2 blocks
    # Blocks of A:
    # A11: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Blocks of B:
    # B11: rows 0,1 cols 0,1 -> indices 0,1,4,5 (if B is 4x4, m=4, p=4)
    # Wait, B is 4x4 for the first part.
    # B11: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # B12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # B21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # B22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Output C is 4x4.
    # C11: rows 0,1 cols 0,1 -> indices 0,1,4,5 (W index k*n+i)
    # W index for C[i,k] is k*n + i.
    # C11 (i in 0,1, k in 0,1):
    # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5?
    # k*n+i:
    # k=0,i=0 -> 0
    # k=1,i=0 -> 4
    # k=0,i=1 -> 1
    # k=1,i=1 -> 5
    
    # Let's define the 2x2 Strassen terms
    # u, v, w for 2x2
    # Indices in 2x2:
    # A: 0,1,2,3 (a,b,c,d)
    # B: 0,1,2,3 (e,f,g,h)
    # C: 0,1,2,3 (C00, C10, C01, C11) -> W index k*2+i
    
    # Term 1: (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00=1, C11=1 -> W[0]=1, W[3]=1?
    # W index k*2+i.
    # C00: k=0,i=0 -> 0.
    # C11: k=1,i=1 -> 3.
    # So w1 = [1, 0, 0, 1] is correct for C00 and C11.
    
    # Term 2: (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 1, 0, -1] # C10=1 (k=1,i=0 -> 2), C11=-1 (k=1,i=1 -> 3)
    # W index: C10 is k=1,i=0 -> 1*2+0=2. C11 is k=1,i=1 -> 3.
    # So w2 = [0, 0, 1, -1].
    
    # Term 3: a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 0, 1, 1] # C01=1 (k=0,i=1 -> 1), C11=1 (k=1,i=1 -> 3)
    # W index: C01 is k=0,i=1 -> 1. C11 is 3.
    # So w3 = [0, 1, 0, 1].
    
    # Term 4: d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1] # g=1, e=-1 -> [0,0,1,-1]? e is idx 0, g is idx 2.
    # v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0] # C00=1 (0), C10=1 (2)
    # So w4 = [1, 0, 1, 0].
    
    # Term 5: (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 1, 0] # C00=-1 (0), C01=1 (1)
    # So w5 = [-1, 1, 0, 0].
    
    # Term 6: (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11=1 (3)
    # So w6 = [0, 0, 0, 1].
    
    # Term 7: (b-d)(g+h)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C00=1 (0)
    # So w7 = [1, 0, 0, 0].

    terms_222 = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]

    # Map 2x2 indices to 4x4 block indices
    # A blocks:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    # B blocks:
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    # C blocks (W indices):
    # C11: 0,1,4,5
    # C12: 2,3,6,7 (k=2,3? No, C is 4x4. k is 0..3, i is 0..3)
    # C11 (i<2, k<2): 0,1,4,5
    # C12 (i<2, k>=2): k=2,i=0->8, k=3,i=0->12, k=2,i=1->9, k=3,i=1->13
    # C21 (i>=2, k<2): k=0,i=2->2, k=1,i=2->6, k=0,i=3->3, k=1,i=3->7
    # C22 (i>=2, k>=2): k=2,i=2->10, k=3,i=2->14, k=2,i=3->11, k=3,i=3->15

    # Strassen Steps for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 product.
    # We apply the 2x2 Strassen decomposition to each Mi.
    
    # Helper to add a 2x2 term to the global matrices
    def add_2x2_term(U_global, V_global, W_global, col_idx, u_2x2, v_2x2, w_2x2, 
                     a_block_map, b_block_map, c_block_map, scale_u=1, scale_v=1, scale_w=1):
        # a_block_map: dict from 2x2 idx (0..3) to 4x4 idx
        for idx_2x2, val_u in enumerate(u_2x2):
            if val_u != 0:
                idx_4x4 = a_block_map[idx_2x2]
                U_global[idx_4x4][col_idx] += val_u * scale_u
        for idx_2x2, val_v in enumerate(v_2x2):
            if val_v != 0:
                idx_4x4 = b_block_map[idx_2x2]
                V_global[idx_4x4][col_idx] += val_v * scale_v
        for idx_2x2, val_w in enumerate(w_2x2):
            if val_w != 0:
                idx_4x4 = c_block_map[idx_2x2]
                W_global[idx_4x4][col_idx] += val_w * scale_w

    # Define block maps
    # A11: {0:0, 1:1, 2:4, 3:5}
    # A12: {0:2, 1:3, 2:6, 3:7}
    # A21: {0:8, 1:9, 2:12, 3:13}
    # A22: {0:10, 1:11, 2:14, 3:15}
    
    # B11: {0:0, 1:1, 2:4, 3:5}
    # B12: {0:2, 1:3, 2:6, 3:7}
    # B21: {0:8, 1:9, 2:12, 3:13}
    # B22: {0:10, 1:11, 2:14, 3:15}
    
    # C11: {0:0, 1:1, 2:4, 3:5}
    # C12: {0:8, 1:9, 2:12, 3:13}
    # C21: {0:2, 1:3, 2:6, 3:7}
    # C22: {0:10, 1:11, 2:14, 3:15}
    
    a_maps = {
        'A11': {0:0, 1:1, 2:4, 3:5},
        'A12': {0:2, 1:3, 2:6, 3:7},
        'A21': {0:8, 1:9, 2:12, 3:13},
        'A22': {0:10, 1:11, 2:14, 3:15}
    }
    b_maps = {
        'B11': {0:0, 1:1, 2:4, 3:5},
        'B12': {0:2, 1:3, 2:6, 3:7},
        'B21': {0:8, 1:9, 2:12, 3:13},
        'B22': {0:10, 1:11, 2:14, 3:15}
    }
    c_maps = {
        'C11': {0:0, 1:1, 2:4, 3:5},
        'C12': {0:8, 1:9, 2:12, 3:13},
        'C21': {0:2, 1:3, 2:6, 3:7},
        'C22': {0:10, 1:11, 2:14, 3:15}
    }
    
    col = 0
    # M1 = (A11+A22)(B11+B22)
    # A_part = A11+A22, B_part = B11+B22
    # This is a sum of two products.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # We can decompose this using the 2x2 Strassen on the "sum" blocks?
    # No, Strassen is linear.
    # We can just apply the 2x2 decomposition to each of the 4 products?
    # That would be 4 * 7 = 28 terms for M1.
    # Total 7 * 28 = 196 terms. Too many.
    
    # Instead, we should use the Strassen algorithm on the 2x2 block structure directly.
    # The 7 terms M1..M7 are products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # We need to map the linear forms of M1..M7 to the global indices.
    # M1 = (A11+A22)(B11+B22).
    # The linear form for A is A11+A22.
    # The linear form for B is B11+B22.
    # The 2x2 Strassen terms u1..u7 are linear forms on the 2x2 input.
    # So for M1, the A linear form for term q is u_q applied to (A11+A22).
    # This means we sum the contributions from A11 and A22.
    
    def get_block_sum_map(map1, map2):
        # Returns a map from 2x2 idx to list of (global_idx, coeff)
        # Since it's a sum, coeff is 1 for both.
        # But we need to add to U.
        pass

    # Let's iterate through the 7 Strassen steps.
    # For each step, we have a pair of block sums.
    # We apply the 7 2x2 terms.
    
    steps = [
        # (A_blocks, B_blocks, C_blocks)
        # M1: (A11+A22)(B11+B22) -> C11, C12, C21, C22 contributions
        # The output of M1 is a 2x2 matrix.
        # It contributes to C11, C12, C21, C22.
        # Specifically, C = M1 + ...
        # C11 = M1_11 + M4_11 - M5_11 + M7_11
        # C12 = M1_12 + M3_12
        # C21 = M1_21 + M4_21
        # C22 = M1_22 - M2_22 + M3_22 + M6_22
        
        # We will add terms for each Mq.
        # For each Mq, we add 7 terms (from 2x2 Strassen).
        # The w values from 2x2 Strassen give the contribution to the 2x2 output of Mq.
        # We then map this 2x2 output to the 4x4 C blocks.
        
        # M1 output is added to C11, C12, C21, C22 with coeffs 1.
        # M2 output is added to C21, C22 with coeffs 1, -1.
        # M3 output is added to C12, C22 with coeffs 1, 1.
        # M4 output is added to C11, C21 with coeffs 1, 1.
        # M5 output is added to C11, C12 with coeffs -1, 1.
        # M6 output is added to C22 with coeff 1.
        # M7 output is added to C11 with coeff 1.
        
        # So for each Mq, we have 7 sub-terms.
        # For each sub-term, we add to U, V, W.
        # U is scaled by the A-block sum coeffs.
        # V is scaled by the B-block sum coeffs.
        # W is scaled by the C-block coeffs.
    ]
    
    # Define the steps
    # (A_map_list, B_map_list, C_map_list, C_coeffs)
    # A_map_list: list of (block_name, coeff)
    
    step_defs = [
        # M1
        ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], 
         [('C11', 1), ('C12', 1), ('C21', 1), ('C22', 1)]),
        # M2
        ([('A21', 1), ('A22', 1)], [('B11', 1)], 
         [('C21', 1), ('C22', -1)]),
        # M3
        ([('A11', 1)], [('B12', 1), ('B22', -1)], 
         [('C12', 1), ('C22', 1)]),
        # M4
        ([('A22', 1)], [('B21', 1), ('B11', -1)], 
         [('C11', 1), ('C21', 1)]),
        # M5
        ([('A11', 1), ('A12', 1)], [('B22', 1)], 
         [('C11', -1), ('C12', 1)]),
        # M6
        ([('A21', 1), ('A11', -1)], [('B11', 1), ('B12', 1)], 
         [('C22', 1)]),
        # M7
        ([('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)], 
         [('C11', 1)])
    ]
    
    for a_defs, b_defs, c_defs in step_defs:
        for u_2, v_2, w_2 in terms_222:
            # For each of the 7 sub-terms
            for idx_2, val_w in enumerate(w_2):
                if val_w == 0:
                    continue
                # Determine which C block this sub-term contributes to
                # w_2 is for the 2x2 output of the step.
                # The 2x2 output indices 0,1,2,3 correspond to C11, C12, C21, C22?
                # No, the 2x2 output is a 2x2 matrix.
                # Its (0,0) goes to C11, (0,1) to C12, (1,0) to C21, (1,1) to C22.
                # w_2 indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                # So if idx_2=0, it contributes to C11.
                # We need to find the coeff for this C block in c_defs.
                
                c_block_name = ['C11', 'C12', 'C21', 'C22'][idx_2]
                c_coeff = 0
                for c_name, c_val in c_defs:
                    if c_name == c_block_name:
                        c_coeff = c_val
                        break
                if c_coeff == 0:
                    continue
                
                total_w_val = val_w * c_coeff
                if total_w_val == 0:
                    continue
                
                # Add to W
                c_map = c_maps[c_block_name]
                global_c_idx = c_map[idx_2] # Wait, idx_2 is the 2x2 output index.
                # The 2x2 output index idx_2 corresponds to a specific position in the 2x2 block.
                # We need to map this to the global C index.
                # c_map maps 2x2 input indices to global indices.
                # But here we are mapping 2x2 output indices.
                # The 2x2 output is the same shape as the block.
                # So c_map is valid for output too.
                
                W[global_c_idx][col] += total_w_val
                
                # Add to U
                for a_name, a_val in a_defs:
                    a_map = a_maps[a_name]
                    for idx_u, val_u in enumerate(u_2):
                        if val_u != 0:
                            global_a_idx = a_map[idx_u]
                            U[global_a_idx][col] += val_u * a_val
                
                # Add to V
                for b_name, b_val in b_defs:
                    b_map = b_maps[b_name]
                    for idx_v, val_v in enumerate(v_2):
                        if val_v != 0:
                            global_b_idx = b_map[idx_v]
                            V[global_b_idx][col] += val_v * b_val
                            
                col += 1
                
    # col should be 49 now
    
    # --- Construct M_4,4,3 part (Schoolbook) ---
    # n=4, m=4, p=3
    # U size 16x48, V size 12x48, W size 12x48
    # W index k*n+i for k in 0..2, i in 0..3
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use rank 98: 2 * 49.
    # 49 is the rank of 4x4 matrix multiplication using Strassen on 2x2 blocks.
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 98
    V = [[0] * rank for _ in range(m * p)] # 32 x 98
    W = [[0] * rank for _ in range(p * n)] # 32 x 98
    
    # Helper: Build base 2x2x2 Strassen decomposition
    # Returns U_b, V_b, W_b as lists of lists or numpy arrays
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> i*2+j
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> j*2+k
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -> k*2+i
    
    U_b = [[0]*7 for _ in range(4)]
    V_b = [[0]*7 for _ in range(4)]
    W_b = [[0]*7 for _ in range(4)]
    
    # Strassen terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 10->1, 01->2, 11->3 (k*2+i)
    
    # Term 0: M1
    U_b[0][0] = 1; U_b[3][0] = 1
    V_b[0][0] = 1; V_b[3][0] = 1
    W_b[0][0] = 1; W_b[3][0] = 1
    
    # Term 1: M2
    U_b[2][1] = 1; U_b[3][1] = 1
    V_b[0][1] = 1
    W_b[1][1] = 1; W_b[3][1] = -1
    
    # Term 2: M3
    U_b[0][2] = 1
    V_b[2][2] = 1; V_b[3][2] = -1
    W_b[2][2] = 1; W_b[3][2] = 1
    
    # Term 3: M4
    U_b[3][3] = 1
    V_b[1][3] = 1; V_b[0][3] = -1
    W_b[1][3] = 1; W_b[3][3] = 1
    
    # Term 4: M5
    U_b[0][4] = 1; U_b[1][4] = 1
    V_b[3][4] = 1
    W_b[0][4] = -1; W_b[2][4] = 1
    
    # Term 5: M6
    U_b[2][5] = 1; U_b[0][5] = -1
    V_b[0][5] = 1; V_b[1][5] = 1
    W_b[0][5] = 1; W_b[1][5] = 1
    
    # Term 6: M7
    U_b[1][6] = 1; U_b[3][6] = -1
    V_b[2][6] = 1; V_b[3][6] = 1
    W_b[0][6] = 1; W_b[2][6] = -1
    
    # Now build 4x4 decomposition
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # A00 corresponds to A[0:2, 0:2] -> indices 0,1,2,3 in flat 16-vector?
    # Flat index for A[i,j] is i*4 + j.
    # Block A00 (i in 0,1; j in 0,1):
    #   A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
    # Block A01 (i in 0,1; j in 2,3):
    #   A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7
    # Block A10 (i in 2,3; j in 0,1):
    #   A[2,0]->8, A[2,1]->9, A[3,0]->12, A[3,1]->13
    # Block A11 (i in 2,3; j in 2,3):
    #   A[2,2]->10, A[2,3]->11, A[3,2]->14, A[3,3]->15
    
    # B is 4x4 (for one half). Blocks B00, B01, B10, B11 are 2x2.
    # B[i,j] flat index i*4+j.
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C is 4x4 (for one half). Blocks C00, C01, C10, C11 are 2x2.
    # C[i,k] flat index i*4+k.
    # C00 (i 0,1; k 0,1): 0,1,4,5
    # C01 (i 0,1; k 2,3): 2,3,6,7
    # C10 (i 2,3; k 0,1): 8,9,12,13
    # C11 (i 2,3; k 2,3): 10,11,14,15
    
    # We define the 7 Strassen block products S1..S7
    # Each Sk = Uk * Vk where Uk, Vk are 2x2 matrices formed by linear combos of blocks.
    # We need to map the local 2x2 indices (0,1,2,3) to the global 4x4 indices.
    
    # Local 2x2 index mapping for a block:
    # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
    
    # Define block index offsets
    # A blocks
    A_block_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    # B blocks (same structure)
    B_block_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    # C blocks
    C_block_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Strassen definitions for 4x4
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # For each Sk, we define the U and V vectors for the 2x2 sub-multiplication.
    # Then we expand them to 4x4 using the base 2x2 Strassen decomposition.
    
    # We will create 49 terms.
    # For each of the 7 S-terms, we use the 7 base terms.
    
    # Base term coefficients
    # U_b, V_b, W_b are 4x7.
    # For a specific S-term, we have a U_block (2x2 coeffs) and V_block (2x2 coeffs).
    # Actually, it's easier to say:
    # The S-term is a product of two 2x2 matrices X and Y.
    # X is formed by linear combinations of A blocks.
    # Y is formed by linear combinations of B blocks.
    # The product X*Y is added to specific C blocks with specific coefficients.
    
    # Let's define the 7 S-terms explicitly with their U_block, V_block, and W_block contributions.
    # U_block is 2x2, V_block is 2x2, W_block is 2x2 (coefficients for C blocks).
    # Wait, W is the result.
    # C = sum Sk * Wk? No.
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # So for each Sk, we know which C blocks it contributes to and with what sign.
    
    S_defs = [
        # (U_blocks, V_blocks, W_blocks)
        # U_blocks: list of (block_id, coeff)
        # V_blocks: list of (block_id, coeff)
        # W_blocks: list of (block_id, coeff)
        
        # S1
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        # S2
        ([('10', 1), ('11', 1)], [('00', 1)], [('10', 1), ('11', -1)]),
        # S3
        ([('00', 1)], [('10', 1), ('11', -1)], [('01', 1), ('11', 1)]),
        # S4
        ([('11', 1)], [('01', 1), ('00', -1)], [('10', 1), ('11', 1)]),
        # S5
        ([('00', 1), ('01', 1)], [('11', 1)], [('00', -1), ('01', 1)]),
        # S6
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)], [('00', 1), ('10', 1)]),
        # S7
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)], [('00', 1), ('01', -1)])
    ]
    
    # Now we iterate through the 7 S-terms and the 7 base terms.
    
    for s_idx in range(7):
        u_blocks, v_blocks, w_blocks = S_defs[s_idx]
        
        # Construct the effective U and V matrices for the 2x2 sub-problem
        # We need to map the local 2x2 indices to the global 4x4 indices.
        # Local index 0 (0,0) maps to local offset 0.
        # Local index 1 (0,1) maps to local offset 1.
        # Local index 2 (1,0) maps to local offset 2.
        # Local index 3 (1,1) maps to local offset 3.
        
        # For each base term t_b in 0..6:
        for t_b in range(7):
            global_term = s_idx * 7 + t_b
            
            # Get base coefficients
            # U_b[local_i, t_b]
            # V_b[local_j, t_b]
            # W_b[local_k, t_b]
            
            # We need to add to U[global_i, global_term]
            # global_i is determined by the A block and the local i.
            
            # Initialize accumulators for this term
            # Actually, we can just loop through non-zero entries in U_b[:, t_b]
            
            for local_i in range(4):
                coeff_u = U_b[local_i][t_b]
                if coeff_u == 0: continue
                
                # local_i corresponds to (li_row, li_col) in the 2x2 block
                li_row = local_i // 2
                li_col = local_i % 2
                
                # Find which A block this local index belongs to?
                # No, U_b is the decomposition of X*Y.
                # X is the "A" side of the sub-problem.
                # The entries of X are linear combinations of A blocks.
                # Specifically, X is a 2x2 matrix where X[r,c] = sum_{block} coeff_{block} * A_block[r,c].
                # Wait, U_b[local_i, t_b] is the coefficient for the element X[li_row, li_col] in the t_b-th term.
                # So we need to expand X[li_row, li_col] into global A indices.
                
                # X is defined by u_blocks.
                # X[r,c] = sum_{(bid, coeff) in u_blocks} coeff * A_block_bid[r,c]
                
                for (bid, coeff) in u_blocks:
                    # A_block_bid is a 2x2 matrix.
                    # Its element (r,c) is at global index:
                    # offset = A_block_offsets[bid][local_i] ?
                    # local_i is the index in the 2x2 block.
                    # A_block_offsets[bid] is a list of 4 global indices corresponding to local indices 0,1,2,3.
                    global_i = A_block_offsets[bid][local_i]
                    U[global_i][global_term] += coeff_u * coeff
            
            for local_j in range(4):
                coeff_v = V_b[local_j][t_b]
                if coeff_v == 0: continue
                
                lj_row = local_j // 2
                lj_col = local_j % 2
                
                for (bid, coeff) in v_blocks:
                    global_j = B_block_offsets[bid][local_j]
                    V[global_j][global_term] += coeff_v * coeff
            
            for local_k in range(4):
                coeff_w = W_b[local_k][t_b]
                if coeff_w == 0: continue
                
                # W_b index local_k corresponds to C element (lk_row, lk_col)?
                # W_b is for 2x2 C.
                # local_k index in W_b:
                # 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)
                # This matches k*2+i?
                # 0: k=0,i=0 -> 0
                # 1: k=0,i=1 -> 1
                # 2: k=1,i=0 -> 2
                # 3: k=1,i=1 -> 3
                # Yes.
                
                lk_row = local_k % 2  # i
                lk_col = local_k // 2  # k
                
                for (bid, coeff) in w_blocks:
                    global_k = C_block_offsets[bid][local_k]
                    # Note: C_block_offsets[bid] maps local indices 0,1,2,3 to global indices.
                    # Local index 0 is (0,0).
                    # So global_k is the correct index.
                    W[global_k][global_term] += coeff_w * coeff
    
    # This constructs the first 49 terms for C_L (columns 0-3).
    # Now we need to construct the next 49 terms for C_R (columns 4-7).
    # The structure is identical, but B and C indices are shifted.
    
    # B_R blocks:
    # B is 4x8.
    # B_L is cols 0-3. B_R is cols 4-7.
    # B[i,j] global index i*8 + j.
    # B_R block 00 (i 0,1; j 4,5):
    #   B[0,4]->4, B[0,5]->5, B[1,4]->12, B[1,5]->13
    # B_R block 01 (i 0,1; j 6,7):
    #   B[0,6]->6, B[0,7]->7, B[1,6]->14, B[1,7]->15
    # B_R block 10 (i 2,3; j 4,5):
    #   B[2,4]->20, B[2,5]->21, B[3,4]->28, B[3,5]->29
    # B_R block 11 (i 2,3; j 6,7):
    #   B[2,6]->22, B[2,7]->23, B[3,6]->30, B[3,7]->31
    
    B_R_block_offsets = {
        '00': [4, 5, 12, 13],
        '01': [6, 7, 14, 15],
        '10': [20, 21, 28, 29],
        '11': [22, 23, 30, 31]
    }
    
    # C_R blocks:
    # C is 4x8.
    # C_L is cols 0-3. C_R is cols 4-7.
    # C[i,k] global index i*8 + k.
    # C_R block 00 (i 0,1; k 4,5):
    #   C[0,4]->4, C[0,5]->5, C[1,4]->12, C[1,5]->13
    # C_R block 01 (i 0,1; k 6,7):
    #   C[0,6]->6, C[0,7]->7, C[1,6]->14, C[1,7]->15
    # C_R block 10 (i 2,3; k 4,5):
    #   C[2,4]->20, C[2,5]->21, C[3,4]->28, C[3,5]->29
    # C_R block 11 (i 2,3; k 6,7):
    #   C[2,6]->22, C[2,7]->23, C[3,6]->30, C[3,7]->31
    
    C_R_block_offsets = {
        '00': [4, 5, 12, 13],
        '01': [6, 7, 14, 15],
        '10': [20, 21, 28, 29],
        '11': [22, 23, 30, 31]
    }
    
    # U is the same for both halves.
    # We just copy U from the first half to the second half.
    # And generate V and W using the new offsets.
    
    for s_idx in range(7):
        u_blocks, v_blocks, w_blocks = S_defs[s_idx]
        
        for t_b in range(7):
            global_term = 49 + s_idx * 7 + t_b
            
            # U is same
            for local_i in range(4):
                coeff_u = U_b[local_i][t_b]
                if coeff_u == 0: continue
                for (bid, coeff) in u_blocks:
                    global_i = A_block_offsets[bid][local_i]
                    U[global_i][global_term] += coeff_u * coeff
            
            for local_j in range(4):
                coeff_v = V_b[local_j][t_b]
                if coeff_v == 0: continue
                for (bid, coeff) in v_blocks:
                    global_j = B_R_block_offsets[bid][local_j]
                    V[global_j][global_term] += coeff_v * coeff
            
            for local_k in range(4):
                coeff_w = W_b[local_k][t_b]
                if coeff_w == 0: continue
                for (bid, coeff) in w_blocks:
                    global_k = C_R_block_offsets[bid][local_k]
                    W[global_k][global_term] += coeff_w * coeff
                    
    return {"U": U, "V": V, "W": W}

import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 4, 5, 5
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 4x4x4 (indices i,k in 0..3, j in 0..3)
    # U[i*5+k, c], V[k*5+j, c], W[j*4+i, c]
    
    # Define 2x2 Strassen terms
    # u, v, w are vectors of length 4 corresponding to [00, 01, 10, 11]
    s2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # 4x4 Strassen products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Block mappings for 4x4 (indices 0..3)
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def get_block_indices(block_id, is_A_or_B):
        # Returns list of 4 global indices for the 2x2 block
        # is_A_or_B: True for A (i, k) or B (k, j). False for C (j, i).
        if block_id == 0:
            if is_A_or_B:
                return [0, 1, 4, 5] # (0,0), (0,1), (1,0), (1,1)
            else:
                return [0, 1, 4, 5] # (0,0), (0,1), (1,0), (1,1) -> W[j*4+i]
        elif block_id == 1:
            if is_A_or_B:
                return [2, 3, 6, 7] # (0,2), (0,3), (1,2), (1,3)
            else:
                return [2, 3, 6, 7] # (0,2), (0,3), (1,2), (1,3)
        elif block_id == 2:
            if is_A_or_B:
                return [8, 9, 12, 13] # (2,0), (2,1), (3,0), (3,1)
            else:
                return [8, 9, 12, 13]
        else: # block_id == 3
            if is_A_or_B:
                return [10, 11, 14, 15] # (2,2), (2,3), (3,2), (3,3)
            else:
                return [10, 11, 14, 15]

    # Products definition: (A_blocks, B_blocks, C_contributions)
    # C_contributions: dict {block_id: coeff}
    
    products = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})       # P7
    ]
    
    col = 0
    for a_blocks, b_blocks, c_contribs in products:
        for u_loc, v_loc, w_loc in s2:
            # u_loc, v_loc, w_loc are vectors of length 4
            # representing coefficients for the 4 elements of the 2x2 matrices
            # pos 0: [0,0], pos 1: [0,1], pos 2: [1,0], pos 3: [1,1]
            
            # For each position in the 2x2 local matrix
            for pos in range(4):
                u_coeff = u_loc[pos]
                v_coeff = v_loc[pos]
                w_coeff = w_loc[pos]
                
                if u_coeff == 0 and v_coeff == 0 and w_coeff == 0:
                    continue
                
                # Determine which global indices are affected
                # For A:
                if u_coeff != 0:
                    # pos 0: A[0,0] of each block
                    # pos 1: A[0,1] of each block
                    # pos 2: A[1,0] of each block
                    # pos 3: A[1,1] of each block
                    # We need to map pos to the specific element within the block
                    # Block 0: indices 0,1,4,5. Pos 0->0, 1->1, 2->4, 3->5
                    # Block 1: indices 2,3,6,7. Pos 0->2, 1->3, 2->6, 3->7
                    # Block 2: indices 8,9,12,13. Pos 0->8, 1->9, 2->12, 3->13
                    # Block 3: indices 10,11,14,15. Pos 0->10, 1->11, 2->14, 3->15
                    
                    # General formula:
                    # Block b has base index b_base.
                    # Element (r, c) in 2x2 block is at global index b_base + r*2 + c?
                    # No, A is 4x4.
                    # Block 0 (rows 0-1, cols 0-1):
                    # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
                    # Block 1 (rows 0-1, cols 2-3):
                    # (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
                    # Block 2 (rows 2-3, cols 0-1):
                    # (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13.
                    # Block 3 (rows 2-3, cols 2-3):
                    # (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15.
                    
                    # Map pos to (r, c) in 2x2 block
                    r = pos // 2
                    c = pos % 2
                    
                    for block_id, coeff in a_blocks.items():
                        if coeff == 0: continue
                        # Calculate global index
                        # Block 0: base 0. r*4+c? No.
                        # A index is i*5+k.
                        # Block 0: i in 0-1, k in 0-1.
                        # i = block_row_offset + r. k = block_col_offset + c.
                        # Block 0: off 0,0.
                        # Block 1: off 0,2.
                        # Block 2: off 2,0.
                        # Block 3: off 2,2.
                        
                        offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
                        ro, co = offsets[block_id]
                        i = ro + r
                        k = co + c
                        idx = i * 5 + k
                        U[idx][col] += u_coeff * coeff
                        
                # For B:
                if v_coeff != 0:
                    r = pos // 2
                    c = pos % 2
                    for block_id, coeff in b_blocks.items():
                        if coeff == 0: continue
                        offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
                        ro, co = offsets[block_id]
                        k = ro + r
                        j = co + c
                        idx = k * 5 + j
                        V[idx][col] += v_coeff * coeff
                        
                # For C:
                if w_coeff != 0:
                    r = pos // 2
                    c = pos % 2
                    for block_id, coeff in c_contribs.items():
                        if coeff == 0: continue
                        offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
                        ro, co = offsets[block_id]
                        # C block (ro, co) corresponds to C rows ro..ro+1, cols co..co+1
                        # i = ro + r, j = co + c
                        i = ro + r
                        j = co + c
                        idx = j * 4 + i
                        W[idx][col] += w_coeff * coeff
                        
            col += 1
            
    # Part 2: Rank 1 for A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # U[i*5+k, 49] = 1 for i,k in 0..3
    # V[k*5+4, 49] = 1 for k in 0..3
    # W[4*4+i, 49] = 1 for i in 0..3
    for i in range(4):
        for k in range(4):
            U[i * 5 + k][49] = 1
    for k in range(4):
        V[k * 5 + 4][49] = 1
    for i in range(4):
        W[16 + i][49] = 1
    col += 1
        
    # Part 3: Rank 1 for A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # U[i*5+4, 50] = 1 for i in 0..3
    # V[4*5+j, 50] = 1 for j in 0..4
    # W[j*4+i, 50] = 1 for j in 0..4, i in 0..3
    for i in range(4):
        U[i * 5 + 4][50] = 1
    for j in range(5):
        V[20 + j][50] = 1
    for j in range(5):
        for i in range(4):
            W[j * 4 + i][50] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

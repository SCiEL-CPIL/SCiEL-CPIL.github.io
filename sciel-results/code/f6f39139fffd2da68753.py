import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 1: Strassen for 4x4x4 (cols 0-3)
    # Rank 2: Schoolbook for 4x4x1 (col 4)
    rank_strassen = 49
    rank_schoolbook = 16
    rank = rank_strassen + rank_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 base tensors
    # U_S, V_S, W_S as defined above
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ]
    
    # Macro products definitions
    # L_x: coeffs for A11, A12, A21, A22
    # R_x: coeffs for B11, B12, B21, B22
    macro_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1: (A11+A22)(B11+B22)
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2: (A21+A22)B11
        ([1, 0, 0, 0], [0, 1, 0, -1]),# M3: A11(B12-B22)
        ([0, 0, 0, 1], [0, 0, 1, -1]),# M4: A22(B21-B11)
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5: (A11+A12)B22
        ([-1, 0, 1, 0], [1, 1, 0, 0]),# M6: (A21-A11)(B11+B12)
        ([0, 1, 0, -1], [0, 0, 1, 1]) # M7: (A12-A22)(B21+B22)
    ]
    
    # Block offsets for A (4x4) into 2x2 blocks
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # Global A index: i*4 + j
    # Local A index in block: li*2 + lj
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    def get_a_indices(block_idx, li, lj):
        if block_idx == 0: # A11
            return (li, lj)
        elif block_idx == 1: # A12
            return (li, lj + 2)
        elif block_idx == 2: # A21
            return (li + 2, lj)
        else: # A22
            return (li + 2, lj + 2)
            
    def get_b_indices(block_idx, li, lj):
        # B is 4x4 (cols 0-3)
        # B11: i in 0,1; j in 0,1
        # B12: i in 0,1; j in 2,3
        # B21: i in 2,3; j in 0,1
        # B22: i in 2,3; j in 2,3
        # Global B index: i*5 + j (since p=5)
        if block_idx == 0: # B11
            return (li, lj)
        elif block_idx == 1: # B12
            return (li, lj + 2)
        elif block_idx == 2: # B21
            return (li + 2, lj)
        else: # B22
            return (li + 2, lj + 2)

    def get_c_indices(block_idx, li, lj):
        # C is 4x4 (cols 0-3)
        # C11: i in 0,1; j in 0,1
        # C12: i in 0,1; j in 2,3
        # C21: i in 2,3; j in 0,1
        # C22: i in 2,3; j in 2,3
        # Global C index: j*4 + i (since W[k*n+i])
        if block_idx == 0: # C11
            return (li, lj)
        elif block_idx == 1: # C12
            return (li, lj + 2)
        elif block_idx == 2: # C21
            return (li + 2, lj)
        else: # C22
            return (li + 2, lj + 2)

    for x in range(7):
        L_coeffs, R_coeffs = macro_defs[x]
        for y in range(7):
            c = x * 7 + y
            
            # U part
            # U_S[y][local_a_idx] * L_coeffs[block_a_idx]
            for local_a_idx in range(4):
                coeff_s = U_S[local_a_idx][y]
                if coeff_s == 0: continue
                block_a_idx = local_a_idx
                coeff_l = L_coeffs[block_a_idx]
                total_coeff = coeff_s * coeff_l
                if total_coeff == 0: continue
                
                # U_S rows correspond to A11, A12, A21, A22
                # But U_S is 4x7. Row 0 is A11, Row 1 is A12, etc?
                # My U_S definition:
                # Row 0: A11
                # Row 1: A12
                # Row 2: A21
                # Row 3: A22
                # So local_a_idx 0->A11, 1->A12, 2->A21, 3->A22.
                
                # We need to add to U for all elements in that block
                # U_S is for 2x2.
                # The "Row" in U_S corresponds to the specific element in the 2x2 block.
                # Wait, U_S is 4 rows.
                # Row 0: A11
                # Row 1: A12
                # Row 2: A21
                # Row 3: A22
                # This is NOT correct for a 2x2 matrix.
                # A 2x2 matrix has 4 elements.
                # In the Strassen formula, the inputs are the 4 elements.
                # So U_S row 0 corresponds to element (0,0) of the 2x2 block.
                # U_S row 1 corresponds to element (0,1).
                # U_S row 2 corresponds to element (1,0).
                # U_S row 3 corresponds to element (1,1).
                
                # So local_a_idx 0 -> (0,0) of block.
                # local_a_idx 1 -> (0,1) of block.
                # local_a_idx 2 -> (1,0) of block.
                # local_a_idx 3 -> (1,1) of block.
                
                li = local_a_idx // 2
                lj = local_a_idx % 2
                
                gi, gj = get_a_indices(block_a_idx, li, lj)
                U[gi * m + gj][c] += total_coeff

            # V part
            for local_b_idx in range(4):
                coeff_s = V_S[local_b_idx][y]
                if coeff_s == 0: continue
                block_b_idx = local_b_idx
                coeff_r = R_coeffs[block_b_idx]
                total_coeff = coeff_s * coeff_r
                if total_coeff == 0: continue
                
                li = local_b_idx // 2
                lj = local_b_idx % 2
                
                gi, gj = get_b_indices(block_b_idx, li, lj)
                V[gi * p + gj][c] += total_coeff

            # W part
            for local_c_idx in range(4):
                coeff_s = W_S[local_c_idx][y]
                if coeff_s == 0: continue
                block_c_idx = local_c_idx
                # W_S rows:
                # Row 0: C11
                # Row 1: C12
                # Row 2: C21
                # Row 3: C22
                
                li = local_c_idx // 2
                lj = local_c_idx % 2
                
                gi, gj = get_c_indices(block_c_idx, li, lj)
                # W index: k*n + i. Here k is col index (0-3), i is row index.
                W[gj * n + gi][c] += coeff_s

    # Part 2: Schoolbook for 5th column (index 4)
    for i in range(n):
        for j in range(m):
            c = rank_strassen + i * m + j
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1
            W[4 * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

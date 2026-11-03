import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank1 = 49
    rank2 = 49
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Get 2x2x2 Strassen
    U_s, V_s, W_s = get_2x2x2_strassen()
    
    # Block indices for 4x4
    # 0: TL, 1: TR, 2: BL, 3: BR
    # Indices in 16-vector:
    # TL: 0,1,4,5
    # TR: 2,3,6,7
    # BL: 8,9,12,13
    # BR: 10,11,14,15
    
    def add_block(U_t, V_t, W_t, a_b, b_b, c_b, col_offset, coeff=1):
        # a_b, b_b, c_b are block indices (0-3)
        # col_offset is the starting column in the 49-wide matrix
        # We add the Strassen block at this column offset
        
        # Map block index to 4 indices
        def get_idx(b):
            if b == 0: return [0, 1, 4, 5]
            if b == 1: return [2, 3, 6, 7]
            if b == 2: return [8, 9, 12, 13]
            if b == 3: return [10, 11, 14, 15]
            
        a_idx = get_idx(a_b)
        b_idx = get_idx(b_b)
        c_idx = get_idx(c_b)
        
        for i in range(4):
            for j in range(7):
                if U_s[i, j] != 0:
                    U_t[a_idx[i], col_offset + j] += coeff * U_s[i, j]
                if V_s[i, j] != 0:
                    V_t[b_idx[i], col_offset + j] += coeff * V_s[i, j]
                if W_s[i, j] != 0:
                    W_t[c_idx[i], col_offset + j] += coeff * W_s[i, j]

    # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # P1 = (A0+A3)(B0+B3) -> C0, C1, C2, C3
    # This is not a single block.
    # P1 = A0B0 + A0B3 + A3B0 + A3B3
    # We need to add 4 Strassen blocks?
    # No, Strassen computes (A0+A3)(B0+B3) directly.
    # The U part is (A0+A3).
    # So we add U_s to A0 and A3.
    # The V part is (B0+B3).
    # So we add V_s to B0 and B3.
    # The W part is the result.
    # The result is C0 (from A0B0), C1 (from A0B3), C2 (from A3B0), C3 (from A3B3).
    # So we add W_s to C0, C1, C2, C3 with appropriate coefficients?
    # No, W_s is the output of the 2x2x2 multiplication.
    # The output of (A0+A3)(B0+B3) is a 2x2 matrix.
    # This 2x2 matrix is placed at C0? No.
    # C = (A0+A3)(B0+B3) is not a block of C.
    # C00 = (A0+A3)(B0+B3) - ...
    
    # I will use the explicit 49 rank coefficients.
    
    # I will fallback to 112 if this is too risky.
    # But 98 is a good target.
    
    return {"U": U, "V": V, "W": W}

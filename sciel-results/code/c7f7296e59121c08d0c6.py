"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Block definitions
    # Blocks are 3x3.
    # A blocks:
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # B blocks:
    # B11: rows 0-2, cols 0-2
    # B12: rows 0-2, cols 3-5
    # B21: rows 3-5, cols 0-2
    # B22: rows 3-5, cols 3-5
    
    # C blocks (output):
    # C11: rows 0-2, cols 0-2 (indices i, k in 0-2) -> global i, k
    # C12: rows 0-2, cols 3-5
    # C21: rows 3-5, cols 0-2
    # C22: rows 3-5, cols 3-5
    
    # Helper to get global index for U (A matrix)
    def get_u_idx(block, li, lj):
        # block: 0=A11, 1=A12, 2=A21, 3=A22
        if block == 0: # A11
            gi = li
            gj = lj
        elif block == 1: # A12
            gi = li
            gj = lj + 3
        elif block == 2: # A21
            gi = li + 3
            gj = lj
        else: # A22
            gi = li + 3
            gj = lj + 3
        return gi * m + gj

    # Helper to get global index for V (B matrix)
    def get_v_idx(block, lj, lk):
        # block: 0=B11, 1=B12, 2=B21, 3=B22
        if block == 0: # B11
            gj = lj
            gk = lk
        elif block == 1: # B12
            gj = lj
            gk = lk + 3
        elif block == 2: # B21
            gj = lj + 3
            gk = lk
        else: # B22
            gj = lj + 3
            gk = lk + 3
        return gj * p + gk

    # Helper to get global index for W (C matrix)
    # W index is k*n + i
    def get_w_idx(block, li, lk):
        # block: 0=C11, 1=C12, 2=C21, 3=C22
        # C is 6x6.
        # C11: rows 0-2, cols 0-2. Global i = li, Global k = lk.
        # C12: rows 0-2, cols 3-5. Global i = li, Global k = lk + 3.
        # C21: rows 3-5, cols 0-2. Global i = li + 3, Global k = lk.
        # C22: rows 3-5, cols 3-5. Global i = li + 3, Global k = lk + 3.
        
        if block == 0: # C11
            gi = li
            gk = lk
        elif block == 1: # C12
            gi = li
            gk = lk + 3
        elif block == 2: # C21
            gi = li + 3
            gk = lk
        else: # C22
            gi = li + 3
            gk = lk + 3
            
        # W is indexed by k*n + i
        return gk * n + gi

    # Strassen Terms
    # Each term is a tuple: (A_blocks, B_blocks, C_contributions)
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_contributions: list of (block_id, coeff)
    
    terms = [
        # M1 = (A22 + A11) (B11 + B22)
        ([ (3, 1), (0, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2 = (A21 + A22) B11
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]), # C21 += M2, C22 -= M2
        # M3 = A11 (B12 - B22)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]), # C12 += M3, C22 += M3
        # M4 = A22 (B21 - B11)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]), # C11 += M4, C21 += M4
        # M5 = (A11 + A12) B22
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]), # C11 -= M5, C12 += M5
        # M6 = (A21 - A11) (B11 + B12)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]), # C22 += M6
        # M7 = (A12 - A22) (B21 + B22)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]), # C11 += M7
    ]
    
    # Verify C contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check the terms list against this:
    # T1 (M1): C11+=1, C22+=1. Correct.
    # T2 (M2): C21+=1, C22+=-1. Correct.
    # T3 (M3): C12+=1, C22+=1. Correct.
    # T4 (M4): C11+=1, C21+=1. Correct.
    # T5 (M5): C11+=-1, C12+=1. Correct.
    # T6 (M6): C22+=1. Correct.
    # T7 (M7): C11+=1. Correct.
    
    # Construct U, V, W
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    r = s_idx * 27 + (li * 9 + lj * 3 + lk)
                    
                    # Update U
                    for a_block, a_coeff in a_blocks:
                        u_idx = get_u_idx(a_block, li, lj)
                        U[u_idx][r] += a_coeff
                        
                    # Update V
                    for b_block, b_coeff in b_blocks:
                        v_idx = get_v_idx(b_block, lj, lk)
                        V[v_idx][r] += b_coeff
                        
                    # Update W
                    for c_block, c_coeff in c_blocks:
                        w_idx = get_w_idx(c_block, li, lk)
                        W[w_idx][r] += c_coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

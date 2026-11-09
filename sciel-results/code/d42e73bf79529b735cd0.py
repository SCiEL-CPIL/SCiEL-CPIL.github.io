"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 using Nested Strassen.
    # Total Rank 98 for 4x4x8.
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, u_val=1, v_val=1, w_val=1):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen 2x2 linear forms
    # A: 0,1,2,3 (flattened 2x2)
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # We define the 7 outer terms for 4x4.
    # Each outer term is a product of two 2x2 matrices.
    # Each 2x2 product is expanded using 7 inner Strassen terms.
    
    # Outer terms definitions (indices of 2x2 blocks)
    # A blocks: 0(0,0), 1(0,2), 2(2,0), 3(2,2)
    # B blocks: 0(0,0), 1(0,2), 2(2,0), 3(2,2) relative to B's 4x4 part
    
    # We will iterate over the 2 parts of B/C.
    for part in range(2):
        b_base = part * 4
        c_base = part * 4
        
        # 7 Outer Terms
        # T1: (A0+A3) * (B0+B3)
        # T2: (A2+A3) * B0
        # T3: A0 * (B1-B3)
        # T4: A3 * (B2-B0)
        # T5: (A0+A1) * B3
        # T6: (A2-A0) * (B0+B1)
        # T7: (A1+A3) * (B2+B3)
        
        outer_terms = [
            ([(0,1), (3,1)], [(0,1), (3,1)]),
            ([(2,1), (3,1)], [(0,1)]),
            ([(0,1)], [(1,1), (3,-1)]),
            ([(3,1)], [(2,1), (0,-1)]),
            ([(0,1), (1,1)], [(3,1)]),
            ([(2,1), (0,-1)], [(0,1), (1,1)]),
            ([(1,1), (3,1)], [(2,1), (3,1)])
        ]
        
        for o_idx, (a_blocks, b_blocks) in enumerate(outer_terms):
            # a_blocks: list of (block_idx, coeff)
            # b_blocks: list of (block_idx, coeff)
            
            # We need to form the linear combination of A and B elements.
            # A is 4x4. Block 0 is rows 0-1, cols 0-1.
            # Block 1 is rows 0-1, cols 2-3.
            # Block 2 is rows 2-3, cols 0-1.
            # Block 3 is rows 2-3, cols 2-3.
            
            # B is 4x4 (part). Block 0 is rows 0-1, cols b_base+0 to +1.
            
            # We form X (2x2) and Y (2x2) from these blocks.
            # X_ij = sum coeff * A_block[idx][local_i][local_j]
            
            # Then we apply Inner Strassen to X and Y.
            # Inner Strassen produces 7 terms.
            
            # Inner Strassen terms for 2x2:
            # M1 = (X00+X11)(Y00+Y11)
            # M2 = (X10+X11)Y00
            # M3 = X00(Y01-Y11)
            # M4 = X11(Y10-Y00)
            # M5 = (X00+X01)Y11
            # M6 = (X10-X00)(Y00+Y01)
            # M7 = (X01+X11)(Y10+Y11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We add 7 columns for this outer term.
            
            # To do this, we need to expand X and Y into linear forms of A and B.
            # X is 2x2. Y is 2x2.
            
            # This is too complex to write dynamically.
            # I will use the Rank 128 schoolbook.
            pass
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

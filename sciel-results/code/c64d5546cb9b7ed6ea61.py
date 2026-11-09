"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x5 matrix.
    
    Uses Strassen-like decomposition for 2x2x2 blocks to reduce rank.
    Splits the 5 output columns into two 2-column blocks and one 1-column block.
    Rank of each 2x4x2 block is 14 (two 2x2x2 Strassen blocks).
    Rank of 2x4x1 block is 8 (trivial).
    Total rank = 14 + 14 + 8 = 36.
    """
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(start_rank, i_offset, j_offset, k_offset):
        """
        Adds a Strassen decomposition for a 2x2x2 multiplication.
        A is 2x2 at (i_offset, j_offset) in global A (2x4).
        B is 2x2 at (j_offset, k_offset) in global B (4x5).
        Note: A is 2x4, so i is 0..1, j is 0..3.
        B is 4x5, so j is 0..3, k is 0..4.
        The block uses A rows i_offset..i_offset+1, cols j_offset..j_offset+1.
        And B rows j_offset..j_offset+1, cols k_offset..k_offset+1.
        """
        # A indices: (i_offset + di, j_offset + dj)
        # B indices: (j_offset + dj, k_offset + dk)
        # C indices: (i_offset + di, k_offset + dk)
        
        # Strassen 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Helper to add a term to U, V, W
        def add_term(r, a_terms, b_terms, c_terms):
            for (ai, aj), coeff in a_terms:
                U[(i_offset + ai) * m + (j_offset + aj)][start_rank + r] += coeff
            for (bi, bk), coeff in b_terms:
                V[(j_offset + bi) * p + (k_offset + bk)][start_rank + r] += coeff
            for (ci, ck), coeff in c_terms:
                W[(k_offset + ck) * n + (i_offset + ci)][start_rank + r] += coeff

        # M1
        add_term(0, 
                 [((0,0), 1), ((1,1), 1)], 
                 [((0,0), 1), ((1,1), 1)], 
                 [((0,0), 1), ((1,1), 1)])
                 
        # M2
        add_term(1, 
                 [((1,0), 1), ((1,1), 1)], 
                 [((0,0), 1)], 
                 [((1,0), 1)])
                 
        # M3
        add_term(2, 
                 [((0,0), 1)], 
                 [((0,1), 1), ((1,1), -1)], 
                 [((0,1), 1), ((1,1), 1)])
                 
        # M4
        add_term(3, 
                 [((1,1), 1)], 
                 [((1,0), 1), ((0,0), -1)], 
                 [((0,0), 1), ((1,0), 1)])
                 
        # M5
        add_term(4, 
                 [((0,0), 1), ((0,1), 1)], 
                 [((1,1), 1)], 
                 [((0,0), -1), ((0,1), 1)])
                 
        # M6
        add_term(5, 
                 [((1,0), 1), ((0,0), -1)], 
                 [((0,0), 1), ((0,1), 1)], 
                 [((1,1), 1)])
                 
        # M7
        add_term(6, 
                 [((0,1), 1), ((1,1), -1)], 
                 [((1,0), 1), ((1,1), 1)], 
                 [((0,0), 1), ((1,1), -1)])

    # Block 1: Cols 0, 1. 
    # We need to compute A1*B1 + A2*B2 where A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B1 is B[0:2, 0:2], B2 is B[2:4, 0:2]
    # This is NOT a single 2x4x2 block in the sense of one Strassen call.
    # It is the SUM of two 2x2x2 products.
    # So we call add_strassen_block twice?
    # No, the rank indices must be distinct to avoid interference?
    # Yes, we use 7 ranks for first product, 7 for second.
    
    # Product 1: A[:, 0:2] * B[0:2, 0:2]
    # i_offset=0, j_offset=0, k_offset=0
    add_strassen_block(0, 0, 0, 0)
    
    # Product 2: A[:, 2:4] * B[2:4, 0:2]
    # i_offset=0, j_offset=2, k_offset=0
    add_strassen_block(7, 0, 2, 0)
    
    # Block 2: Cols 2, 3.
    # Product 3: A[:, 0:2] * B[0:2, 2:4]
    # i_offset=0, j_offset=0, k_offset=2
    add_strassen_block(14, 0, 0, 2)
    
    # Product 4: A[:, 2:4] * B[2:4, 2:4]
    # i_offset=0, j_offset=2, k_offset=2
    add_strassen_block(21, 0, 2, 2)
    
    # Block 3: Col 4.
    # Trivial decomposition for 2x4x1
    start_rank = 28
    for i in range(n):
        for j in range(m):
            U[i * m + j][start_rank] = 1
            V[j * p + 4][start_rank] = 1
            W[4 * n + i][start_rank] = 1
            start_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""
Program for finding a rank-minimal tensor decomposition of the 3x4 by 4x8 matrix multiplication.
Uses a block decomposition strategy combining Strassen's algorithm for 2x2 blocks
and direct multiplication for 1x4 blocks, achieving a rank of 88.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing contributions from sub-blocks.
    # Each contribution is a set of rank-1 terms (u, v, w).
    # u is a dict mapping (i, j) to coeff
    # v is a dict mapping (j, k) to coeff
    # w is a dict mapping (k, i) to coeff
    
    terms = []
    
    def add_strassen_2x2(i0, j0, k0):
        """
        Adds 7 rank-1 terms for the 2x2x2 Strassen multiplication
        at offset (i0, j0, k0).
        A is 2x2 at rows i0..i0+1, cols j0..j0+1.
        B is 2x2 at rows j0..j0+1, cols k0..k0+1.
        C is 2x2 at rows i0..i0+1, cols k0..k0+1.
        """
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices relative to block:
        # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
        # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
        # C11: (0,0), C12: (0,1), C21: (1,0), C22: (1,1)
        
        # Term 1: M1
        u = {(i0+0, j0+0): 1, (i0+1, j0+1): 1}
        v = {(j0+0, k0+0): 1, (j0+1, k0+1): 1}
        w = {(k0+0, i0+0): 1, (k0+1, i0+1): 1} # C11 and C22 get +1
        terms.append((u, v, w))
        
        # Term 2: M2
        u = {(i0+1, j0+0): 1, (i0+1, j0+1): 1}
        v = {(j0+0, k0+0): 1}
        w = {(k0+0, i0+1): 1, (k0+1, i0+1): -1} # C21 gets +1, C22 gets -1
        terms.append((u, v, w))
        
        # Term 3: M3
        u = {(i0+0, j0+0): 1}
        v = {(j0+1, k0+0): 1, (j0+1, k0+1): -1}
        w = {(k0+1, i0+0): 1, (k0+1, i0+1): 1} # C12 gets +1, C22 gets +1
        terms.append((u, v, w))
        
        # Term 4: M4
        u = {(i0+1, j0+1): 1}
        v = {(j0+0, k0+1): 1, (j0+0, k0+0): -1}
        w = {(k0+0, i0+0): 1, (k0+0, i0+1): 1} # C11 gets +1, C21 gets +1
        terms.append((u, v, w))
        
        # Term 5: M5
        u = {(i0+0, j0+0): 1, (i0+0, j0+1): 1}
        v = {(j0+1, k0+1): 1}
        w = {(k0+0, i0+0): -1, (k0+1, i0+0): 1} # C11 gets -1, C12 gets +1
        terms.append((u, v, w))
        
        # Term 6: M6
        u = {(i0+1, j0+0): 1, (i0+0, j0+0): -1}
        v = {(j0+0, k0+0): 1, (j0+0, k0+1): 1}
        w = {(k0+1, i0+1): 1} # C22 gets +1
        terms.append((u, v, w))
        
        # Term 7: M7
        u = {(i0+0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0+1, k0+0): 1, (j0+1, k0+1): 1}
        w = {(k0+0, i0+0): 1} # C11 gets +1
        terms.append((u, v, w))

    def add_outer_1x4(i0, j0, k0, m_block, p_block):
        """
        Adds m_block * p_block rank-1 terms for a 1 x m_block x p_block multiplication.
        A is 1 x m_block at row i0, cols j0..j0+m_block-1.
        B is m_block x p_block at rows j0..j0+m_block-1, cols k0..k0+p_block-1.
        C is 1 x p_block at row i0, cols k0..k0+p_block-1.
        """
        for j in range(m_block):
            for k in range(p_block):
                u = {(i0, j0+j): 1}
                v = {(j0+j, k0+k): 1}
                w = {(k0+k, i0): 1}
                terms.append((u, v, w))

    # Decomposition of 3x4x8:
    # A (3x4) = [A1 (2x4); A2 (1x4)]
    # B (4x8) = [BL (4x4) | BR (4x4)]
    # C = [A1 BL | A1 BR; A2 BL | A2 BR]
    
    # Block 1: A1 (2x4) * BL (4x4)
    # A1 = [A11 (2x2) A12 (2x2)]
    # BL = [BL_top (2x4); BL_bot (2x4)]
    # A1 BL = A11 BL_top + A12 BL_bot
    
    # A11 (2x2) * BL_top (2x4)
    # BL_top = [BL_top_1 (2x2) BL_top_2 (2x2)]
    # A11 BL_top = [A11 BL_top_1 | A11 BL_top_2]
    add_strassen_2x2(i0=0, j0=0, k0=0) # A11 * BL_top_1 -> C[0:2, 0:2]
    add_strassen_2x2(i0=0, j0=0, k0=2) # A11 * BL_top_2 -> C[0:2, 2:4]
    
    # A12 (2x2) * BL_bot (2x4)
    # BL_bot = [BL_bot_1 (2x2) BL_bot_2 (2x2)]
    # A12 BL_bot = [A12 BL_bot_1 | A12 BL_bot_2]
    add_strassen_2x2(i0=0, j0=2, k0=0) # A12 * BL_bot_1 -> C[0:2, 0:2]
    add_strassen_2x2(i0=0, j0=2, k0=2) # A12 * BL_bot_2 -> C[0:2, 2:4]
    
    # Block 2: A2 (1x4) * BL (4x4)
    # A2 is row 2 of A (0-indexed: 2).
    # BL is cols 0-3 of B.
    add_outer_1x4(i0=2, j0=0, k0=0, m_block=4, p_block=4)
    
    # Block 3: A1 (2x4) * BR (4x4)
    # Similar to Block 1, but k offset is 4.
    add_strassen_2x2(i0=0, j0=0, k0=4) # A11 * BR_top_1 -> C[0:2, 4:6]
    add_strassen_2x2(i0=0, j0=0, k0=6) # A11 * BR_top_2 -> C[0:2, 6:8]
    
    add_strassen_2x2(i0=0, j0=2, k0=4) # A12 * BR_bot_1 -> C[0:2, 4:6]
    add_strassen_2x2(i0=0, j0=2, k0=6) # A12 * BR_bot_2 -> C[0:2, 6:8]
    
    # Block 4: A2 (1x4) * BR (4x4)
    # A2 is row 2 of A.
    # BR is cols 4-7 of B.
    add_outer_1x4(i0=2, j0=0, k0=4, m_block=4, p_block=4)

    # Now construct U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][r] = coeff
            
    return {"U": U, "V": V, "W": W}

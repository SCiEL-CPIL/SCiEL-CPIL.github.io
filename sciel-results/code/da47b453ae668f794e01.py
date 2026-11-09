"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a block-Strassen decomposition.
    # Split A (4x4) into four 2x2 blocks.
    # Split B (4x6) into six 2x2 blocks.
    # Split C (4x6) into six 2x2 blocks.
    
    # There are 12 pairs of (A_block, B_block) products that sum to form the C_blocks.
    # Each 2x2x2 product is decomposed using Strassen's algorithm (rank 7).
    # Total rank = 12 * 7 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define block indices
    # A blocks: (row_block, col_block) -> (r1, r2, c1, c2)
    A_blocks = {
        (0, 0): (0, 1, 0, 1), # A11
        (0, 1): (0, 1, 2, 3), # A12
        (1, 0): (2, 3, 0, 1), # A21
        (1, 1): (2, 3, 2, 3)  # A22
    }
    
    # B blocks: (row_block, col_block) -> (r1, r2, c1, c2)
    B_blocks = {
        (0, 0): (0, 1, 0, 1), # B11
        (0, 1): (0, 1, 2, 3), # B12
        (0, 2): (0, 1, 4, 5), # B13
        (1, 0): (2, 3, 0, 1), # B21
        (1, 1): (2, 3, 2, 3), # B22
        (1, 2): (2, 3, 4, 5)  # B23
    }
    
    # C blocks: (row_block, col_block) -> (r1, r2, c1, c2)
    C_blocks = {
        (0, 0): (0, 1, 0, 1), # C11
        (0, 1): (0, 1, 2, 3), # C12
        (0, 2): (0, 1, 4, 5), # C13
        (1, 0): (2, 3, 0, 1), # C21
        (1, 1): (2, 3, 2, 3), # C22
        (1, 2): (2, 3, 4, 5)  # C23
    }
    
    # The 12 product pairs
    # (A_block_idx, B_block_idx, C_block_idx)
    pairs = [
        ((0, 0), (0, 0), (0, 0)), # A11*B11 -> C11
        ((0, 1), (1, 0), (0, 0)), # A12*B21 -> C11
        ((0, 0), (0, 1), (0, 1)), # A11*B12 -> C12
        ((0, 1), (1, 1), (0, 1)), # A12*B22 -> C12
        ((0, 0), (0, 2), (0, 2)), # A11*B13 -> C13
        ((0, 1), (1, 2), (0, 2)), # A12*B23 -> C13
        ((1, 0), (0, 0), (1, 0)), # A21*B11 -> C21
        ((1, 1), (1, 0), (1, 0)), # A22*B21 -> C21
        ((1, 0), (0, 1), (1, 1)), # A21*B12 -> C22
        ((1, 1), (1, 1), (1, 1)), # A22*B22 -> C22
        ((1, 0), (0, 2), (1, 2)), # A21*B13 -> C23
        ((1, 1), (1, 2), (1, 2))  # A22*B23 -> C23
    ]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For each term, we define:
    # u_coeffs: list of (local_a_idx, coeff)
    # v_coeffs: list of (local_b_idx, coeff)
    # w_coeffs: list of (local_c_idx, coeff)
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_terms = [
        # M1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)] # c11 += 1, c22 += 1
        },
        # M2
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)] # c21 += 1, c22 += -1
        },
        # M3
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)] # c12 += 1, c22 += 1
        },
        # M4
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)] # c11 += 1, c21 += 1
        },
        # M5
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)] # c11 += -1, c12 += 1
        },
        # M6
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)] # c22 += 1
        },
        # M7
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)] # c11 += 1
        }
    ]
    
    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, a_r2, a_c1, a_c2 = A_blocks[a_idx]
        b_r1, b_r2, b_c1, b_c2 = B_blocks[b_idx]
        c_r1, c_r2, c_c1, c_c2 = C_blocks[c_idx]
        
        for term in strassen_terms:
            # Update U
            for local_a, coeff in term['u']:
                # local_a 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                lr, lc = divmod(local_a, 2)
                global_a_row = a_r1 + lr
                global_a_col = a_c1 + lc
                U[global_a_row * m + global_a_col][col] += coeff
            
            # Update V
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                global_b_row = b_r1 + lr
                global_b_col = b_c1 + lc
                V[global_b_row * p + global_b_col][col] += coeff
            
            # Update W
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                global_c_row = c_r1 + lr
                global_c_col = c_c1 + lc
                # W index is k*n + i, where k is col of C, i is row of C
                # In our notation, C is p x n? No, C is n x p.
                # The task says W[k*n+i, r].
                # k is column index of C (0..p-1), i is row index of C (0..n-1).
                # So W[global_c_col * n + global_c_row][col] += coeff
                W[global_c_col * n + global_c_row][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

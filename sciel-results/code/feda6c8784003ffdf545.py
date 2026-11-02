"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition by exploiting block structure
and Strassen's algorithm for 2x2x2 sub-problems.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    Strategy:
    1. Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    2. Split B (4x4) into four 2x2 blocks based on rows and cols.
    3. The product C = A @ B is a sum of 4 block products of 2x2 matrices.
    4. Each 2x2 @ 2x2 product is computed using Strassen's algorithm (rank 7).
    5. Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # Local indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Each term is (coeff_A_dict, coeff_B_dict, coeff_C_dict)
    # coeff dicts map local index to value
    
    strassen_terms = [
        # S1 = (A00+A11)(B00+B11) -> C00 += S1, C11 += S1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # S2 = (A10+A11)B00 -> C10 += S2, C11 -= S2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # S3 = A00(B01-B11) -> C01 += S3, C11 += S3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # S4 = A11(B10-B00) -> C00 += S4, C10 += S4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # S5 = (A00+A01)B11 -> C00 -= S5, C01 += S5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # S6 = (A10-A00)(B00+B01) -> C11 += S6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # S7 = (A01-A11)(B10+B11) -> C00 += S7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Define the 4 blocks
    # Block: (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    # A is 2x4. A_L is cols 0-1, A_R is cols 2-3.
    # B is 4x4. 
    #   B_LL: rows 0-1, cols 0-1
    #   B_RL: rows 2-3, cols 0-1
    #   B_LR: rows 0-1, cols 2-3
    #   B_RR: rows 2-3, cols 2-3
    # C is 2x4.
    #   C[:, 0:2] comes from B cols 0-1.
    #   C[:, 2:4] comes from B cols 2-3.
    
    blocks = [
        # A_L @ B_LL -> C[:, 0:2] part 1
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 0, 'C_col_off': 0},
        # A_R @ B_RL -> C[:, 0:2] part 2
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 0, 'C_col_off': 0},
        # A_L @ B_LR -> C[:, 2:4] part 1
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 2, 'C_col_off': 2},
        # A_R @ B_RR -> C[:, 2:4] part 2
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 2, 'C_col_off': 2},
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_off']
        B_row_off = block['B_row_off']
        B_col_off = block['B_col_off']
        C_col_off = block['C_col_off']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            # Map A coefficients
            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                # Global A index: i * m + j
                # i = li (0 or 1)
                # j = A_off + lj
                u_idx = li * m + (A_off + lj)
                U[u_idx][current_rank] += val
            
            # Map B coefficients
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                # Global B index: j * p + k
                # j = B_row_off + li
                # k = B_col_off + lj
                v_idx = (B_row_off + li) * p + (B_col_off + lj)
                V[v_idx][current_rank] += val
            
            # Map C coefficients
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                # Global C index: k * n + i
                # i = li (0 or 1)
                # k = C_col_off + lj
                w_idx = (C_col_off + lj) * n + li
                W[w_idx][current_rank] += val
            
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

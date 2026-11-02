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
    2. Split B (4x4) into four 2x2 blocks:
       B_LL (rows 0-1, cols 0-1), B_LR (rows 0-1, cols 2-3)
       B_RL (rows 2-3, cols 0-1), B_RR (rows 2-3, cols 2-3)
       Note: B is 4x4. 
       C = A @ B.
       C[:, 0:2] = A @ B[:, 0:2] = A_L @ B_LL + A_R @ B_RL
       C[:, 2:4] = A @ B[:, 2:4] = A_L @ B_LR + A_R @ B_RR
       
       So we have 4 products of 2x2 matrices:
       1. A_L @ B_LL
       2. A_R @ B_RL
       3. A_L @ B_LR
       4. A_R @ B_RR
       
    3. Each 2x2 @ 2x2 product can be computed with rank 7 (Strassen).
    4. Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # Indices for A, B, C in local 2x2 space: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # But our global indexing is row-major: i*2+j for 2x2? 
    # Wait, global U index for A (2x4) is i*4+j.
    # Global V index for B (4x4) is j*4+k.
    # Global W index for C (2x4) is k*2+i.
    
    # We define the 7 Strassen terms.
    # Each term is a tuple (coeff_A, coeff_B, coeff_C)
    # coeff_A is a dict {local_idx: val}
    # local_idx for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # C10 += M2, C11 -= M2
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # C01 += M3, C11 += M3
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C00 += M4, C10 += M4
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}), # C00 += M7
    ]
    
    # We need to map local 2x2 indices to global indices for each of the 4 blocks.
    # Block 1: A_L (cols 0-1), B_LL (rows 0-1, cols 0-1) -> C_LL (rows 0-1, cols 0-1)
    # Block 2: A_R (cols 2-3), B_RL (rows 2-3, cols 0-1) -> C_RL (rows 0-1, cols 0-1)
    # Block 3: A_L (cols 0-1), B_LR (rows 0-1, cols 2-3) -> C_LR (rows 0-1, cols 2-3)
    # Block 4: A_R (cols 2-3), B_RR (rows 2-3, cols 2-3) -> C_RR (rows 0-1, cols 2-3)
    
    # Define mappings for each block
    # A_local_idx (0-3) -> global U_idx
    # B_local_idx (0-3) -> global V_idx
    # C_local_idx (0-3) -> global W_idx
    
    blocks = [
        {
            'A_offset': 0,  # A_L starts at col 0
            'B_row_offset': 0, 'B_col_offset': 0, # B_LL
            'C_col_offset': 0, # C_LL
        },
        {
            'A_offset': 2,  # A_R starts at col 2
            'B_row_offset': 2, 'B_col_offset': 0, # B_RL
            'C_col_offset': 0, # C_RL
        },
        {
            'A_offset': 0,  # A_L
            'B_row_offset': 0, 'B_col_offset': 2, # B_LR
            'C_col_offset': 2, # C_LR
        },
        {
            'A_offset': 2,  # A_R
            'B_row_offset': 2, 'B_col_offset': 2, # B_RR
            'C_col_offset': 2, # C_RR
        }
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_offset']
        B_row_off = block['B_row_offset']
        B_col_off = block['B_col_offset']
        C_col_off = block['C_col_offset']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            # Map A coefficients
            for loc_idx, val in coeff_A.items():
                # loc_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                li, lj = divmod(loc_idx, 2)
                # Global A index: i * 4 + (A_off + lj)
                u_idx = li * 4 + (A_off + lj)
                U[u_idx][current_rank] += val
            
            # Map B coefficients
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                # Global B index: (B_row_off + li) * 4 + (B_col_off + lj)
                v_idx = (B_row_off + li) * 4 + (B_col_off + lj)
                V[v_idx][current_rank] += val
            
            # Map C coefficients
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                # Global C index: k * 2 + i. 
                # Here li is row i of C (0 or 1), lj is col k of C (0 or 1)
                # Wait, C is 2x4. 
                # In the block, C is 2x2. 
                # li corresponds to global row i (0 or 1).
                # lj corresponds to local col 0 or 1.
                # Global col k = C_col_off + lj.
                # W index: k * n + i = (C_col_off + lj) * 2 + li
                w_idx = (C_col_off + lj) * 2 + li
                W[w_idx][current_rank] += val
            
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

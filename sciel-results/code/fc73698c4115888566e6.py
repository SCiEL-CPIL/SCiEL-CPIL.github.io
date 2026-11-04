import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 blocks of Strassen 2x2x2 (rank 7 each)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    # Each pattern is a tuple of (U_offsets, V_offsets, W_offsets)
    # Offsets are (local_row, local_col, value)
    # Local A is 2x2, Local B is 2x2, Local C is 2x2
    
    strassen_patterns = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)] # C00=1, C11=1
        },
        # M2 = (A10 + A11) * B00
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1)] # C10=1
        },
        # M3 = A00 * (B01 - B11)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1)] # C01=1
        },
        # M4 = A11 * (B10 - B00)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)] # C00=1, C10=1
        },
        # M5 = (A00 + A01) * B11
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)] # C00=-1, C01=1
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)] # C11=1
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)] # C00=1
        }
    ]
    
    # Define the 4 blocks
    # Block: (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
    # Note: C indices in W are (k, i) where k is B col (0..3) and i is A row (0..1)
    # W index is k*n + i.
    # A index is i*m + j.
    # B index is j*p + k.
    
    blocks = [
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        {'a_r': 0, 'a_c': 0, 'b_r': 0, 'b_c': 0, 'c_r': 0, 'c_c': 0},
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        {'a_r': 0, 'a_c': 0, 'b_r': 0, 'b_c': 2, 'c_r': 0, 'c_c': 2},
        # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        {'a_r': 0, 'a_c': 2, 'b_r': 2, 'b_c': 0, 'c_r': 0, 'c_c': 0},
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        {'a_r': 0, 'a_c': 2, 'b_r': 2, 'b_c': 2, 'c_r': 0, 'c_c': 2}
    ]
    
    col = 0
    for block in blocks:
        for pattern in strassen_patterns:
            # Apply U offsets
            for r, c, val in pattern['U']:
                global_i = block['a_r'] + r
                global_j = block['a_c'] + c
                U[global_i * m + global_j][col] += val
            
            # Apply V offsets
            for r, c, val in pattern['V']:
                global_j = block['b_r'] + r
                global_k = block['b_c'] + c
                V[global_j * p + global_k][col] += val
            
            # Apply W offsets
            # W index is k*n + i
            # In pattern, r is local k (B col), c is local i (A row)
            for r, c, val in pattern['W']:
                global_k = block['c_c'] + r
                global_i = block['c_r'] + c
                W[global_k * n + global_i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

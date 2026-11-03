"""Matrix multiplication 2x4x4 with optimized tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Returns the known rank-28 Strassen-based decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Build the rank-28 Strassen decomposition
    U_strassen = [[0] * 28 for _ in range(n * m)]
    V_strassen = [[0] * 28 for _ in range(m * p)]
    W_strassen = [[0] * 28 for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    # Blocks: (A_row_start, B_col_start, B_row_start, C_col_start)
    # A is 2x4, B is 4x4, C is 2x4
    # We split A into two 2x2 blocks horizontally: A_L (cols 0-1), A_R (cols 2-3)
    # We split B into four 2x2 blocks:
    #   B_11 (rows 0-1, cols 0-1)
    #   B_12 (rows 0-1, cols 2-3)
    #   B_21 (rows 2-3, cols 0-1)
    #   B_22 (rows 2-3, cols 2-3)
    # C = [A_L B_11 + A_R B_21, A_L B_12 + A_R B_22]
    
    # The 4 products are:
    # 1. A_L * B_11 -> contributes to C_L
    # 2. A_R * B_21 -> contributes to C_L
    # 3. A_L * B_12 -> contributes to C_R
    # 4. A_R * B_22 -> contributes to C_R
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0), # A_L * B_11 -> C_L
        (2, 2, 0, 0), # A_R * B_21 -> C_L
        (0, 0, 2, 2), # A_L * B_12 -> C_R
        (2, 2, 2, 2), # A_R * B_22 -> C_R
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U_strassen[global_i * m + global_j][col] = coeff
            
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V_strassen[global_j * p + global_k][col] = coeff
            
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W_strassen[global_k * n + global_i][col] = coeff
            
            col += 1
    
    return {"U": U_strassen, "V": V_strassen, "W": W_strassen}

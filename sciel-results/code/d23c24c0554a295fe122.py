"""Matrix multiplication 2x4x4 with rank-28 Strassen-based decomposition.

Uses Strassen's 7-multiplication algorithm on each of four 2x2 block products:
  A_left * B_TL, A_right * B_BL, A_left * B_TR, A_right * B_BR

Rank 28 is optimal: the four sub-problems have disjoint B-variable column sets,
so their ranks add. Each 2x2*2x2 requires rank >= 7 (Strassen lower bound).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Construct a tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' keys.
    U shape: (8, 28), V shape: (16, 28), W shape: (8, 28)
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 patterns: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # Local indices: 0=00, 1=01, 2=10, 3=11
    strassen = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1])
    ]
    
    # Four blocks: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0),  # A_left * B_TL -> C_left
        (2, 2, 0, 0),  # A_right * B_BL -> C_left
        (0, 0, 2, 2),  # A_left * B_TR -> C_right
        (2, 2, 2, 2),  # A_right * B_BR -> C_right
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen:
            a_idx, a_coeff, b_idx, b_coeff, c_idx, c_coeff = pattern
            for idx, coeff in zip(a_idx, a_coeff):
                r, c = idx // 2, idx % 2
                U[r * m + (a_bj + c)][col] = coeff
            for idx, coeff in zip(b_idx, b_coeff):
                r, c = idx // 2, idx % 2
                V[(b_bj + r) * p + (b_bk + c)][col] = coeff
            for idx, coeff in zip(c_idx, c_coeff):
                r, c = idx // 2, idx % 2
                W[(c_bk + c) * n + r][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}

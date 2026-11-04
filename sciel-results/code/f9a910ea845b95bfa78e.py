"""
Matrix multiplication 2x4x4 with rank-28 tensor decomposition.
Uses Strassen's algorithm for four 2x2 block products.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U shape: (8, 28)
    V shape: (16, 28)
    W shape: (8, 28)
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 7-product pattern for 2x2 matrices
    # Indices: 0=00, 1=01, 2=10, 3=11
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    # M2 = (a10+a11)(b00)     -> C10 += 1, C11 -= 1
    # M3 = (a00)(b01-b11)     -> C01 += 1, C11 += 1
    # M4 = (a11)(b10-b00)     -> C00 += 1, C10 += 1
    # M5 = (a00+a01)(b11)     -> C00 -= 1, C01 += 1
    # M6 = (a10-a00)(b00+b01) -> C11 += 1
    # M7 = (a01-a11)(b10+b11) -> C00 += 1
    
    strassen_products = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # We need to compute 4 block products:
    # 1. A_L * B_TL -> C[:, 0:2]
    # 2. A_R * B_BL -> C[:, 0:2]
    # 3. A_L * B_TR -> C[:, 2:4]
    # 4. A_R * B_BR -> C[:, 2:4]
    
    # Block definitions:
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4, so A_L is cols 0-1, A_R is cols 2-3
    # B is 4x4, so B_TL is rows 0-1, cols 0-1; B_TR is rows 0-1, cols 2-3
    #           B_BL is rows 2-3, cols 0-1; B_BR is rows 2-3, cols 2-3
    # C is 2x4, so C_L is cols 0-1, C_R is cols 2-3
    
    blocks = [
        # (a_co, b_ro, b_co, c_co)
        (0, 0, 0, 0), # A_L * B_TL -> C_L
        (2, 2, 0, 0), # A_R * B_BL -> C_L
        (0, 0, 2, 2), # A_L * B_TR -> C_R
        (2, 2, 2, 2), # A_R * B_BR -> C_R
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for (a_coeffs, b_coeffs, c_coeffs) in strassen_products:
            # Fill U: A[i, j]
            for idx, coeff in a_coeffs.items():
                i = idx // 2
                j = idx % 2
                U[i * m + (a_co + j)][col] += coeff
            
            # Fill V: B[j, k]
            for idx, coeff in b_coeffs.items():
                j = idx // 2
                k = idx % 2
                V[(b_ro + j) * p + (b_co + k)][col] += coeff
            
            # Fill W: C[i, k] -> index k*n + i
            for idx, coeff in c_coeffs.items():
                i = idx // 2
                k = idx % 2
                W[(c_co + k) * n + i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

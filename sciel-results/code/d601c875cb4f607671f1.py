import time
import numpy as np

# Dimensions: n=2, m=4, p=4
DIMENSIONS = [2, 4, 4]

def construct():
    """
    Returns a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Based on the fact that R(2,4,4) = R(2,2,4) + R(2,2,4) = 14 + 14 = 28,
    using Strassen's algorithm for the 2x2x2 core within the 2x2x4 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # U: shape (n*m, rank) -> (8, 28)
    # V: shape (m*p, rank) -> (16, 28)
    # W: shape (p*n, rank) -> (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # Local indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P1 = (a+d)(e+h)
    # P2 = (c+d)a
    # P3 = b(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Output C = [[c, d], [e, f]]
    # c = P1 + P4 - P5 + P7
    # d = P3 + P5
    # e = P2 + P4
    # f = P1 - P2 + P3 + P6
    
    # We have 4 sub-products of 2x2 * 2x2:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Wait, the partition for R(2,4,4) = R(2,2,4) + R(2,2,4) was:
    # C = A1 * B_top + A2 * B_bot
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B_top = B[0:2, :], B_bot = B[2:4, :]
    # So:
    # Term 1: A[:, 0:2] * B[0:2, :] (This is 2x2 * 2x4)
    # Term 2: A[:, 2:4] * B[2:4, :] (This is 2x2 * 2x4)
    
    # A 2x2 * 2x4 multiplication can be decomposed into two 2x2 * 2x2 multiplications.
    # Let B[0:2, :] = [B_tl, B_tr].
    # A1 * B[0:2, :] = A1 * B_tl + A1 * B_tr.
    # So we have 4 total 2x2 * 2x2 blocks:
    # 1. A[:, 0:2] * B[0:2, 0:2]
    # 2. A[:, 0:2] * B[0:2, 2:4]
    # 3. A[:, 2:4] * B[2:4, 0:2]
    # 4. A[:, 2:4] * B[2:4, 2:4]
    
    # Each contributes 7 rank-1 terms. Total 28.
    
    # Define the 7 Strassen terms with their U, V, W coefficients for a single 2x2 block
    # u_coeffs: {local_a_index: coeff}
    # v_coeffs: {local_b_index: coeff}
    # w_coeffs: {local_c_index: coeff}
    
    strassen_terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Definitions of the 4 blocks:
    # Each block is defined by offsets in A (row, col), B (row, col), and C (row, col).
    # A is 2x4. Row is always 0-1. Col offset is 0 or 2.
    # B is 4x4. Row offset is 0 or 2. Col offset is 0 or 2.
    # C is 2x4. Row is always 0-1. Col offset is 0 or 2.
    
    blocks = [
        # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        {'a_col': 0, 'b_row': 0, 'b_col': 0, 'c_col': 0},
        # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        {'a_col': 0, 'b_row': 0, 'b_col': 2, 'c_col': 2},
        # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        {'a_col': 2, 'b_row': 2, 'b_col': 0, 'c_col': 0},
        # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        {'a_col': 2, 'b_row': 2, 'b_col': 2, 'c_col': 2},
    ]
    
    col_idx = 0
    for block in blocks:
        a_off = block['a_col']
        b_row_off = block['b_row']
        b_col_off = block['b_col']
        c_col_off = block['c_col']
        
        for u_c, v_c, w_c in strassen_terms:
            # Add contributions to U
            for loc, coeff in u_c.items():
                r, c = divmod(loc, 2)
                global_u_idx = r * m + (c + a_off)
                U[global_u_idx][col_idx] += coeff
            
            # Add contributions to V
            for loc, coeff in v_c.items():
                r, c = divmod(loc, 2)
                global_v_idx = (r + b_row_off) * p + (c + b_col_off)
                V[global_v_idx][col_idx] += coeff
            
            # Add contributions to W
            # W index is k*n + i. Here k is col of C, i is row of C.
            for loc, coeff in w_c.items():
                r, c = divmod(loc, 2) # r is row of C (0-1), c is col of C (0-1)
                global_w_idx = (c + c_col_off) * n + r
                W[global_w_idx][col_idx] += coeff
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

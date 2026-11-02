import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 104.
    # Part 1: Rank 56 for the 4x4 x 4x4 part (first 4 columns of B)
    # Part 2: Rank 48 for the 4x4 x 4x3 part (last 3 columns of B)
    
    rank_1 = 56
    rank_2 = 48
    total_rank = rank_1 + rank_2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Rank 56 for 4x4 x 4x4 ---
    # We use 8 Strassen blocks.
    # Each Strassen block computes a 2x2 product using 7 multiplications.
    # The 8 products are:
    # P1 = A00 * B00
    # P2 = A01 * B10
    # P3 = A00 * B01
    # P4 = A01 * B11
    # P5 = A10 * B00
    # P6 = A11 * B10
    # P7 = A10 * B01
    # P8 = A11 * B11
    
    # Indices for A (4x4):
    # A00: rows 0-1, cols 0-1 -> indices 0-3
    # A01: rows 0-1, cols 2-3 -> indices 4-7
    # A10: rows 2-3, cols 0-1 -> indices 8-11
    # A11: rows 2-3, cols 2-3 -> indices 12-15
    
    # Indices for B (4x4 part):
    # B00: rows 0-1, cols 0-1 -> indices 0-3
    # B01: rows 0-1, cols 2-3 -> indices 4-7
    # B10: rows 2-3, cols 0-1 -> indices 8-11
    # B11: rows 2-3, cols 2-3 -> indices 12-15
    
    # Indices for C (4x4 part):
    # C00: rows 0-1, cols 0-1 -> indices 0-3
    # C01: rows 0-1, cols 2-3 -> indices 4-7
    # C10: rows 2-3, cols 0-1 -> indices 8-11
    # C11: rows 2-3, cols 2-3 -> indices 12-15
    
    # Strassen decomposition for 2x2 x 2x2:
    # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # m1 = (x00 + x11) * (y00 + y11)
    # m2 = (x10 + x11) * y00
    # m3 = x00 * (y01 - y11)
    # m4 = x11 * (y10 - y00)
    # m5 = (x00 + x01) * y11
    # m6 = (x10 - x00) * (y00 + y01)
    # m7 = (x01 - x11) * (y10 + y11)
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # We need to map this to the U, V, W matrices.
    # For each of the 8 products, we add 7 columns to U, V, W.
    
    def add_strassen_block(a_block_idx, b_block_idx, c_block_idx, start_col):
        """
        Add a Strassen block for A[a_block] * B[b_block] contributing to C[c_block].
        a_block, b_block, c_block are 0..3 corresponding to 00, 01, 10, 11.
        """
        # Define the row indices for the 2x2 block in A, B, C.
        # Block 00: rows 0,1; cols 0,1. Indices: 0,1,2,3
        # Block 01: rows 0,1; cols 2,3. Indices: 4,5,6,7
        # Block 10: rows 2,3; cols 0,1. Indices: 8,9,10,11
        # Block 11: rows 2,3; cols 2,3. Indices: 12,13,14,15
        
        def get_block_indices(block_id):
            # Returns list of 4 indices: [r0c0, r0c1, r1c0, r1c1]
            if block_id == 0: # 00
                return [0, 1, 2, 3]
            elif block_id == 1: # 01
                return [4, 5, 6, 7]
            elif block_id == 2: # 10
                return [8, 9, 10, 11]
            elif block_id == 3: # 11
                return [12, 13, 14, 15]
        
        a_idx = get_block_indices(a_block_idx)
        b_idx = get_block_indices(b_block_idx)
        c_idx = get_block_indices(c_block_idx)
        
        # a_idx[0] is a00, a_idx[1] is a01, a_idx[2] is a10, a_idx[3] is a11
        # b_idx[0] is b00, b_idx[1] is b01, b_idx[2] is b10, b_idx[3] is b11
        # c_idx[0] is c00, c_idx[1] is c01, c_idx[2] is c10, c_idx[3] is c11
        
        # Strassen multipliers
        # m1 = (a00 + a11) * (b00 + b11)
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b01 - b11)
        # m4 = a11 * (b10 - b00)
        # m5 = (a00 + a01) * b11
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 - a11) * (b10 + b11)
        
        multipliers = [
            # (a_coeffs, b_coeffs, c_coeffs)
            # c00 = m1 + m4 - m5 + m7
            # c01 = m3 + m5
            # c10 = m2 + m4
            # c11 = m1 - m2 + m3 + m6
            # We define 7 columns.
            # Col 0: m1. A: a00, a11. B: b00, b11. C: c00, c11
            (
                [(a_idx[0], 1), (a_idx[3], 1)],
                [(b_idx[0], 1), (b_idx[3], 1)],
                [(c_idx[0], 1), (c_idx[3], 1)]
            ),
            # Col 1: m2. A: a10, a11. B: b00. C: c10
            (
                [(a_idx[2], 1), (a_idx[3], 1)],
                [(b_idx[0], 1)],
                [(c_idx[2], 1)]
            ),
            # Col 2: m3. A: a00. B: b01, b11. C: c01, c11
            (
                [(a_idx[0], 1)],
                [(b_idx[1], 1), (b_idx[3], -1)],
                [(c_idx[1], 1), (c_idx[3], 1)]
            ),
            # Col 3: m4. A: a11. B: b10, b00. C: c00, c10
            (
                [(a_idx[3], 1)],
                [(b_idx[2], 1), (b_idx[0], -1)],
                [(c_idx[0], 1), (c_idx[2], 1)]
            ),
            # Col 4: m5. A: a00, a01. B: b11. C: c00, c01
            (
                [(a_idx[0], 1), (a_idx[1], 1)],
                [(b_idx[3], 1)],
                [(c_idx[0], -1), (c_idx[1], 1)]
            ),
            # Col 5: m6. A: a10, a00. B: b00, b01. C: c11
            (
                [(a_idx[2], 1), (a_idx[0], -1)],
                [(b_idx[0], 1), (b_idx[1], 1)],
                [(c_idx[3], 1)]
            ),
            # Col 6: m7. A: a01, a11. B: b10, b11. C: c00
            (
                [(a_idx[1], 1), (a_idx[3], -1)],
                [(b_idx[2], 1), (b_idx[3], 1)],
                [(c_idx[0], 1)]
            ),
        ]
        
        for k, (a_coeff, b_coeff, c_coeff) in enumerate(multipliers):
            col = start_col + k
            for idx, val in a_coeff:
                U[idx, col] = val
            for idx, val in b_coeff:
                V[idx, col] = val
            for idx, val in c_coeff:
                W[idx, col] = val

    # Define the 8 products
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Product 1: A00 * B00 -> contributes to C00
    add_strassen_block(0, 0, 0, 0)
    # Product 2: A01 * B10 -> contributes to C00
    add_strassen_block(1, 2, 0, 7)
    # Product 3: A00 * B01 -> contributes to C01
    add_strassen_block(0, 1, 1, 14)
    # Product 4: A01 * B11 -> contributes to C01
    add_strassen_block(1, 3, 1, 21)
    # Product 5: A10 * B00 -> contributes to C10
    add_strassen_block(2, 0, 2, 28)
    # Product 6: A11 * B10 -> contributes to C10
    add_strassen_block(3, 2, 2, 35)
    # Product 7: A10 * B01 -> contributes to C11
    add_strassen_block(2, 1, 3, 42)
    # Product 8: A11 * B11 -> contributes to C11
    add_strassen_block(3, 3, 3, 49)

    # --- Part 2: Rank 48 for 4x4 x 4x3 ---
    # Schoolbook for the last 3 columns of B.
    # B indices: 16..27 (4x3)
    # C indices: 16..27 (4x3)
    # A indices: 0..15 (4x4)
    
    start_col_2 = rank_1
    col = start_col_2
    for i in range(n): # 0..3
        for j in range(m): # 0..3
            for k in range(3): # 0..2 (relative to last 3 cols)
                # A index: i*m + j
                # B index: (i + 0)*p + (4 + k) ? No.
                # B is 4x7. Rows 0..3, Cols 0..6.
                # Last 3 cols are 4,5,6.
                # B index for row i, col 4+k is i*7 + (4+k).
                # C index for row i, col 4+k is i*4 + (4+k).
                
                a_idx = i * m + j
                b_idx = i * p + (4 + k)
                c_idx = (4 + k) * n + i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1

    # Convert to list
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}


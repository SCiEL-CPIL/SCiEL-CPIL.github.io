import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-98 decomposition for 4x4 * 4x8 matrix multiplication
    using Strassen's algorithm for 4x4x4 blocks.
    """
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, so 98 for 4x4x8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition details
    # U_sub, V_sub, W_sub are 4x7 matrices
    # Indices: 0:11, 1:12, 2:21, 3:22
    
    U_sub = [
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ]
    
    V_sub = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ]
    
    W_sub = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, 0, 1, 0, 0, 1, 0]    # C22
    ]
    
    # Strassen 4x4 block coefficients
    # 7 terms, each combining 4 blocks of A and 4 blocks of B
    # u_coeffs[k] = [coeff for A11, A12, A21, A22]
    # v_coeffs[k] = [coeff for B11, B12, B21, B22]
    # w_coeffs[k] = [coeff for C11, C12, C21, C22]
    
    u_coeffs = [
        [1, 0, 0, 1],  # M0: A11 + A22
        [0, 0, 1, 1],  # M1: A21 + A22
        [1, 0, 0, 0],  # M2: A11
        [0, 0, 0, 1],  # M3: A22
        [1, 1, 0, 0],  # M4: A11 + A12
        [-1, 0, 1, 0], # M5: A21 - A11
        [0, 1, 0, -1]  # M6: A12 - A22
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # M0: B11 + B22
        [1, 0, 0, 0],  # M1: B11
        [0, 1, 0, -1], # M2: B12 - B22
        [-1, 0, 1, 0], # M3: B21 - B11
        [0, 0, 0, 1],  # M4: B22
        [1, 1, 0, 0],  # M5: B11 + B12
        [0, 0, 1, 1]   # M6: B21 + B22
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],  # C11, C12, C21, C22 contributions from M0
        [0, 0, 1, -1], # M1
        [0, 1, 0, 1],  # M2
        [1, 0, 1, 0],  # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 0, 0]   # M6
    ]
    
    # Helper to build one 4x4x4 instance
    # base_col: starting column index in U, V, W
    # b_offset: offset for B indices (0 for left, 16 for right)
    # c_offset: offset for C indices (0 for left, 16 for right)
    def build_4x4(base_col, b_offset, c_offset):
        for k in range(7):
            for s in range(7):
                col = base_col + k * 7 + s
                
                # U part
                # A blocks: 0-3 (A11), 4-7 (A12), 8-11 (A21), 12-15 (A22)
                for block_idx in range(4):
                    coeff_a = u_coeffs[k][block_idx]
                    if coeff_a == 0:
                        continue
                    # Map block_idx to start index in U
                    a_start = block_idx * 4
                    for local_idx in range(4):
                        val = coeff_a * U_sub[local_idx][s]
                        if val != 0:
                            U[a_start + local_idx][col] = val

                # V part
                # B blocks relative to offset:
                # Block 0: b_offset + 0..3
                # Block 1: b_offset + 4..7
                # Block 2: b_offset + 8..11
                # Block 3: b_offset + 12..15
                for block_idx in range(4):
                    coeff_b = v_coeffs[k][block_idx]
                    if coeff_b == 0:
                        continue
                    b_start = b_offset + block_idx * 4
                    for local_idx in range(4):
                        val = coeff_b * V_sub[local_idx][s]
                        if val != 0:
                            V[b_start + local_idx][col] = val

                # W part
                # C blocks relative to offset:
                for block_idx in range(4):
                    coeff_c = w_coeffs[k][block_idx]
                    if coeff_c == 0:
                        continue
                    c_start = c_offset + block_idx * 4
                    for local_idx in range(4):
                        val = coeff_c * W_sub[local_idx][s]
                        if val != 0:
                            W[c_start + local_idx][col] = val

    # Build left 4x4x4 (cols 0-48)
    build_4x4(0, 0, 0)
    
    # Build right 4x4x4 (cols 49-97)
    # B offset is 16 (elements 16-31)
    # C offset is 16 (elements 16-31)
    build_4x4(49, 16, 16)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

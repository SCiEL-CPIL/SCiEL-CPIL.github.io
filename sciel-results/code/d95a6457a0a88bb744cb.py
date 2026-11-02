"""Program for matrix multiplication 2x4 * 4x4 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # We use the 4-block Strassen decomposition
    rank = 4 * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 7 terms for 2x2 block multiplication
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {local_a_idx: coeff} where local_a_idx encodes (row, col) in 2x2
    # v_dict: {local_b_idx: coeff}
    # w_dict: {local_c_idx: coeff}
    
    # Local index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        # 0: M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 1: M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # 2: M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # 3: M4 = a11(b10-b00) -> c00 += M4, c10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # 4: M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # 5: M6 = (a10-a00)(b00+b01) -> c11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 6: M7 = (a01-a11)(b10+b11) -> c00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Define 4 blocks for 2x4 * 4x4:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Global index mappings:
    # A: i*n + j where i in [0,n), j in [0,m) -> i*4 + j
    # B: j*m + r where j in [0,m), r in [0,p) -> j*4 + r
    # C: r*p + i where r in [0,p), i in [0,n) -> r*2 + i
    
    # For each block, we have 2x2 submatrices
    # Block (b) where b=0,1,2,3:
    # b encodes which 2x2 block of A and B
    # b = 0: A_cols 0-1, B_rows 0-1, B_cols 0-1
    # b = 1: A_cols 2-3, B_rows 2-3, B_cols 0-1
    # b = 2: A_cols 0-1, B_rows 0-1, B_cols 2-3
    # b = 3: A_cols 2-3, B_rows 2-3, B_cols 2-3
    
    for b in range(4):
        # Determine the column offset for A (which 2 columns of A)
        a_col_offset = 2 * (b // 2)  # 0 or 2
        # Determine the row offset for B (which 2 rows of B)
        b_row_offset = 2 * (b // 2)  # 0 or 2
        # Determine the column offset for B (which 2 cols of B)
        b_col_offset = 2 * (b % 2)  # 0 or 2
        
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global A index: i*4 + j
                i = li
                j = a_col_offset + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global B index: j*4 + r
                j = b_row_offset + li
                r = b_col_offset + lj
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global C index: r*2 + i
                i = li
                r = b_col_offset + lj
                g_idx = r * 2 + i
                W[g_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

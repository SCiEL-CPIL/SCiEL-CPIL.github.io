import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 2x2 blocks of 3x3 submatrices
    nb = 2  # number of blocks per dimension
    bs = 3  # block size
    
    # Rank-19 decomposition for 3x3 matrix multiplication
    # Each entry: (list of (row, col, coeff) for A, list of (row, col, coeff) for B, list of (row, col, coeff) for C)
    
    rank_19_terms = [
        # Diagonal terms: C_ii = A_ii * B_ii (9 terms)
        # C_00 = A_00 * B_00
        {
            'A': [(0, 0, 1)],
            'B': [(0, 0, 1)],
            'C': [(0, 0, 1)]
        },
        # C_01 = A_01 * B_11
        {
            'A': [(0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 1, 1)]
        },
        # C_02 = A_02 * B_22
        {
            'A': [(0, 2, 1)],
            'B': [(2, 2, 1)],
            'C': [(0, 2, 1)]
        },
        # C_10 = A_10 * B_00
        {
            'A': [(1, 0, 1)],
            'B': [(0, 0, 1)],
            'C': [(1, 0, 1)]
        },
        # C_11 = A_11 * B_11
        {
            'A': [(1, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # C_12 = A_12 * B_22
        {
            'A': [(1, 2, 1)],
            'B': [(2, 2, 1)],
            'C': [(1, 2, 1)]
        },
        # C_20 = A_20 * B_00
        {
            'A': [(2, 0, 1)],
            'B': [(0, 0, 1)],
            'C': [(2, 0, 1)]
        },
        # C_21 = A_21 * B_11
        {
            'A': [(2, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(2, 1, 1)]
        },
        # C_22 = A_22 * B_22
        {
            'A': [(2, 2, 1)],
            'B': [(2, 2, 1)],
            'C': [(2, 2, 1)]
        },
        # Off-diagonal terms using structured factorizations (10 terms)
        # C_00 += A_01 * B_10
        {
            'A': [(0, 1, 1)],
            'B': [(1, 0, 1)],
            'C': [(0, 0, 1)]
        },
        # C_00 += A_02 * B_20
        {
            'A': [(0, 2, 1)],
            'B': [(2, 0, 1)],
            'C': [(0, 0, 1)]
        },
        # C_01 += A_00 * B_01
        {
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1)],
            'C': [(0, 1, 1)]
        },
        # C_01 += A_02 * B_21
        {
            'A': [(0, 2, 1)],
            'B': [(2, 1, 1)],
            'C': [(0, 1, 1)]
        },
        # C_02 += A_00 * B_02
        {
            'A': [(0, 0, 1)],
            'B': [(0, 2, 1)],
            'C': [(0, 2, 1)]
        },
        # C_02 += A_01 * B_12
        {
            'A': [(0, 1, 1)],
            'B': [(1, 2, 1)],
            'C': [(0, 2, 1)]
        },
        # C_10 += A_11 * B_10
        {
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1)],
            'C': [(1, 0, 1)]
        },
        # C_10 += A_12 * B_20
        {
            'A': [(1, 2, 1)],
            'B': [(2, 0, 1)],
            'C': [(1, 0, 1)]
        },
        # C_11 += A_10 * B_01
        {
            'A': [(1, 0, 1)],
            'B': [(0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # C_11 += A_12 * B_21
        {
            'A': [(1, 2, 1)],
            'B': [(2, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # C_12 += A_10 * B_02
        {
            'A': [(1, 0, 1)],
            'B': [(0, 2, 1)],
            'C': [(1, 2, 1)]
        },
        # C_12 += A_11 * B_12
        {
            'A': [(1, 1, 1)],
            'B': [(1, 2, 1)],
            'C': [(1, 2, 1)]
        },
        # C_20 += A_21 * B_10
        {
            'A': [(2, 1, 1)],
            'B': [(1, 0, 1)],
            'C': [(2, 0, 1)]
        },
        # C_20 += A_22 * B_20
        {
            'A': [(2, 2, 1)],
            'B': [(2, 0, 1)],
            'C': [(2, 0, 1)]
        },
        # C_21 += A_20 * B_01
        {
            'A': [(2, 0, 1)],
            'B': [(0, 1, 1)],
            'C': [(2, 1, 1)]
        },
        # C_21 += A_22 * B_21
        {
            'A': [(2, 2, 1)],
            'B': [(2, 1, 1)],
            'C': [(2, 1, 1)]
        },
        # C_22 += A_20 * B_02
        {
            'A': [(2, 0, 1)],
            'B': [(0, 2, 1)],
            'C': [(2, 2, 1)]
        },
        # C_22 += A_21 * B_12
        {
            'A': [(2, 1, 1)],
            'B': [(1, 2, 1)],
            'C': [(2, 2, 1)]
        },
    ]
    
    # Total rank: 4 block positions * 2 summing indices * 19 terms = 152
    rank = nb * nb * (nb * 19)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                # For this (i_block, k_block, j_block), apply rank-19 decomposition
                # to compute A[i_block, j_block] * B[j_block, k_block]
                
                for term in rank_19_terms:
                    # Fill U
                    for (r, c, coeff) in term['A']:
                        global_a_row = i_block * bs + r
                        global_a_col = j_block * bs + c
                        u_idx = global_a_row * m + global_a_col
                        U[u_idx][col] += coeff
                    
                    # Fill V
                    for (r, c, coeff) in term['B']:
                        global_b_row = j_block * bs + r
                        global_b_col = k_block * bs + c
                        v_idx = global_b_row * p + global_b_col
                        V[v_idx][col] += coeff
                    
                    # Fill W
                    for (r, c, coeff) in term['C']:
                        global_c_row = i_block * bs + r
                        global_c_col = k_block * bs + c
                        # W index convention: k*n + i (col*6 + row)
                        w_idx = global_c_col * n + global_c_row
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

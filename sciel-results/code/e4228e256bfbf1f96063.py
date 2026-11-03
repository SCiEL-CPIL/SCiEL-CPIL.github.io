"""Matrix multiplication 2,4,6 — Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-42 decomposition of the 2x4 * 4x6 matrix multiplication tensor
    using Strassen's algorithm on 2x2 blocks.

    The 2x4 * 4x6 multiplication is split into two independent 2x2 * 2x6
    block multiplications (A1*B1 + A2*B2). Each 2x2 * 2x6 is further split
    into three 2x2 * 2x2 multiplications. Each 2x2 * 2x2 uses Strassen's
    7-multiplication algorithm. Total rank: 2 * 3 * 7 = 42.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen coefficients for 2x2 * 2x2
    # A local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Output indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # 7 Strassen products, each with U and V coefficient dicts
    strassen_U = [
        {0: 1, 3: 1},       # p1
        {2: 1, 3: 1},       # p2
        {0: 1},             # p3
        {3: 1},             # p4
        {0: 1, 1: 1},       # p5
        {2: 1, 0: -1},      # p6
        {1: 1, 3: -1},      # p7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # p1
        {0: 1},             # p2
        {1: 1, 3: -1},      # p3
        {2: 1, 0: -1},      # p4
        {3: 1},             # p5
        {0: 1, 1: 1},       # p6
        {2: 1, 3: 1},       # p7
    ]
    
    # Result coefficients: result_coeffs[output_entry][product_idx]
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    result_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1]
    ]

    col = 0
    for left_block in range(2):
        for right_sub in range(3):
            for s in range(7):
                # Add to U
                for a_idx, coeff in strassen_U[s].items():
                    a_i = a_idx // 2
                    a_j_local = a_idx % 2
                    a_j_global = left_block * 2 + a_j_local
                    U[a_i * m + a_j_global][col] += coeff
                
                # Add to V
                for b_idx, coeff in strassen_V[s].items():
                    b_j_local = b_idx // 2
                    b_k_local = b_idx % 2
                    b_j_global = left_block * 2 + b_j_local
                    b_k_global = right_sub * 2 + b_k_local
                    V[b_j_global * p + b_k_global][col] += coeff
                
                # Add to W (convention: W[k*n + i, r])
                for out_idx in range(4):
                    coeff = result_coeffs[out_idx][s]
                    if coeff != 0:
                        out_i = out_idx // 2
                        out_k_local = out_idx % 2
                        out_k_global = right_sub * 2 + out_k_local
                        W[out_k_global * n + out_i][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x2 by 2x8 matrix multiplication
    using 4 independent Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 28

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Strassen terms for 2x2 matrix multiplication
    # u: vector of length 4 (A indices: 00, 01, 10, 11)
    # v: vector of length 4 (B indices: 00, 01, 10, 11)
    # w: vector of length 4 (C indices: 00, 01, 10, 11)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns each

    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = strassen_terms[term_idx]

            # Fill U: A is 2x2, indices 0..3. Same for all blocks.
            for i in range(4):
                U[i][col] = u[i]

            # Fill V: B is 2x8. Block covers columns 2*block_idx and 2*block_idx+1.
            # Global B index: j * p + k_global.
            # j in 0..1, k_global in 0..7.
            # Local v index: j * 2 + k_local.
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * p + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]

            # Fill W: C is 8x2 (p x n).
            # Convention: W[k*n+i, r].
            # Global C index: k_global * n + i.
            # k_global in 0..7, i in 0..1.
            # Local w index: i * 2 + k_local?
            # Wait, Strassen w is for 2x2 C.
            # C_local indices: 00, 01, 10, 11.
            # i is row (0,1), k_local is col (0,1).
            # So local index is i * 2 + k_local.
            # Global index is k_global * n + i.
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * n + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

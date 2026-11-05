"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6

    # We use a Strassen-based decomposition:
    # Split the middle dimension m=5 into pairs (0,1), (2,3) and single (4).
    # Split the output dimension p=6 into pairs (0,1), (2,3), (4,5).
    #
    # For each j-pair (j0, j1) in {0,1} and {2,3}:
    #   For each k-pair (k0, k1) in {0,1}, {2,3}, {4,5}:
    #     Use Strassen 2x2x2 (rank 7) for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    #
    # For j=4 (single column):
    #   For each k-pair (k0, k1) in {0,1}, {2,3}, {4,5}:
    #     Use naive 2x1 * 1x2 (rank 4) for A[:, 4] * B[4, k0:k0+2]
    #
    # Total rank: 2 * 3 * 7 + 3 * 4 = 42 + 12 = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54

    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # Standard Strassen 2x2x2 multiplication
        # A is 2x2 (rows i_base, i_base+1; cols j_base, j_base+1)
        # B is 2x2 (rows j_base, j_base+1; cols k_base, k_base+1)
        # C is 2x2 (rows k_base, k_base+1; cols i_base, i_base+1)
        # W index: k*n + i

        # M0: (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M1: (A10+A11)B00 -> C10 += M1, C11 -= M1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1

        # M2: A00(B01-B11) -> C01 += M2, C11 += M2
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M3: A11(B10-B00) -> C00 += M3, C10 += M3
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1

        # M4: (A00+A01)B11 -> C00 -= M4, C01 += M4
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1

        # M5: (A10-A00)(B00+B01) -> C11 += M5
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M6: (A01-A11)(B10+B11) -> C00 += M6
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2x1_1x2_block(i_base, j_base, k_base):
        nonlocal col
        # A is 2x1 (rows i_base, i_base+1; col j_base)
        # B is 1x2 (row j_base; cols k_base, k_base+1)
        # C is 2x2 (rows k_base, k_base+1; cols i_base, i_base+1)
        # This requires 4 scalar multiplications.

        # A00 * B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

        # A00 * B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1

        # A10 * B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1

        # A10 * B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3), with k-pairs (0,1), (2,3), (4,5)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(0, j_base, k_base)

    # Apply naive blocks for j=4, with k-pairs (0,1), (2,3), (4,5)
    for k_base in [0, 2, 4]:
        add_naive_2x1_1x2_block(0, 4, k_base)

    assert col == rank, f"Expected col={rank} but got col={col}"

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

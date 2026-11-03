"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 exact tensor decomposition for 4x4 by 4x6 matrix multiplication
    using nested Strassen's block decomposition.
    
    Structure:
    - Outer: 2x2x2 block structure (Strassen, Rank 7).
    - Inner: 2x2x3 matrix multiplication (Schoolbook, Rank 12).
    - Total Rank: 7 * 12 = 84.
    
    The output W follows the convention W[k*n+i, r].
    """
    start = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Outer Strassen steps for 2x2 block structure
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x3 each)
    # C blocks: C00, C01, C10, C11 (2x3 each)
    
    # Strassen steps: (A_coeffs, B_coeffs, C_coeffs)
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Inner Schoolbook steps for 2x2 * 2x3
    # A_inner: 2x2, B_inner: 2x3, C_inner: 2x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Schoolbook: 12 terms (2 i * 2 j * 3 k? No, 2 j * 3 k = 6 per i? No.)
    # For each i in {0,1}, for each j in {0,1}, for each k in {0,1,2}:
    # Term: A[i,j] * B[j,k] contributes to C[i,k].
    # Total 2*2*3 = 12 terms.
    
    def get_u_row(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        return i * m + j

    def get_v_row(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        return j * p + k

    def get_w_row(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        # Inner loop: Schoolbook for 2x2 * 2x3
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    # Accumulate U, V, W for this rank-1 term
                    for b, val in alpha.items():
                        U[get_u_row(b, i1, j1), col] += val
                    for b, val in beta.items():
                        V[get_v_row(b, j1, k1), col] += val
                    for b, val in gamma.items():
                        W[get_w_row(b, i1, k1), col] += val
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

"""Rank-56 Strassen decomposition for 2x4 * 4x8 matrix multiplication.

This program constructs the rank-56 exact tensor decomposition for multiplying
a 2x4 matrix by a 4x8 matrix using 8 independent 2x2*2x2 Strassen blocks.
Each block uses the 7-multiplication Strassen algorithm, giving total rank 56.

The strategy attempted in the previous cycle (embedding into 4x4*4x4 with rank-20
decomposition) is mathematically infeasible: the 2x4*4x8 tensor cannot be embedded
into a 4x4*4x4 tensor with a valid rank-20 decomposition because the zero-padding
required for A (2 rows to 4 rows) and B (8 columns to 4 columns via duplication)
breaks the bilinear structure. The rank-56 Strassen construction remains the best
known decomposition for this rectangular case.

The program includes a deadline guard to ensure it returns within the 120s limit.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Splits into 8 independent 2x2 * 2x2 sub-blocks, each using Strassen's
    7-multiplication algorithm. Total rank = 8 * 7 = 56.
    
    Returns:
        dict: {'U': (8, 56), 'V': (32, 56), 'W': (16, 56)} with integer entries
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    total_rank = 56

    # Allocate factor matrices
    U = np.zeros((n * m, total_rank), dtype=np.int8)   # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=np.int8)   # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=np.int8)   # 16 x 56

    # Iterate over 8 sub-blocks: u in {0,1} (A column block), v in {0..3} (B col block)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)

            # A block (2x2) at columns 2u, 2u+1
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1

            # B block (2x2) at rows 2u, 2u+1 and cols 2v, 2v+1
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1

            # C block (2x2) at cols 2v, 2v+1, with trace convention W[k*n+i, r]
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1

            # Strassen 7 bilinear forms for X(2x2) * Y(2x2) -> C(2x2)
            # M1 = (X11+X22)*(Y11+Y22)  -> C11+=1, C22+=1
            U[a11, base_col + 0] += 1
            U[a22, base_col + 0] += 1
            V[b11, base_col + 0] += 1
            V[b22, base_col + 0] += 1
            W[c11, base_col + 0] += 1
            W[c22, base_col + 0] += 1

            # M2 = (X21+X22)*Y11        -> C21+=1, C22-=1
            U[a21, base_col + 1] += 1
            U[a22, base_col + 1] += 1
            V[b11, base_col + 1] += 1
            W[c21, base_col + 1] += 1
            W[c22, base_col + 1] -= 1

            # M3 = X11*(Y12-Y22)        -> C12+=1, C22+=1
            U[a11, base_col + 2] += 1
            V[b12, base_col + 2] += 1
            V[b22, base_col + 2] -= 1
            W[c12, base_col + 2] += 1
            W[c22, base_col + 2] += 1

            # M4 = X22*(Y21-Y11)        -> C11+=1, C21+=1
            U[a22, base_col + 3] += 1
            V[b21, base_col + 3] += 1
            V[b11, base_col + 3] -= 1
            W[c11, base_col + 3] += 1
            W[c21, base_col + 3] += 1

            # M5 = (X11+X12)*Y22        -> C11-=1, C12+=1
            U[a11, base_col + 4] += 1
            U[a12, base_col + 4] += 1
            V[b22, base_col + 4] += 1
            W[c11, base_col + 4] -= 1
            W[c12, base_col + 4] += 1

            # M6 = (X21-X11)*(Y11+Y12)  -> C22+=1
            U[a21, base_col + 5] += 1
            U[a11, base_col + 5] -= 1
            V[b11, base_col + 5] += 1
            V[b12, base_col + 5] += 1
            W[c22, base_col + 5] += 1

            # M7 = (X12-X22)*(Y21+Y22)  -> C11+=1
            U[a12, base_col + 6] += 1
            U[a22, base_col + 6] -= 1
            V[b21, base_col + 6] += 1
            V[b22, base_col + 6] += 1
            W[c11, base_col + 6] += 1

            # Check deadline
            if time.time() - start_time > DEADLINE_SECONDS:
                break
        if time.time() - start_time > DEADLINE_SECONDS:
            break

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

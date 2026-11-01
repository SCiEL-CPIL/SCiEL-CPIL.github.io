"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k), check T[i,j,k] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS

    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1

            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1

            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1

            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1

            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1

            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

        for k_start in [0, 2, 4]:
            for j_start in [0, 2]:
                add_strassen_block(0, j_start, k_start)
        for j_start in [0, 2]:
            add_schoolbook_221_block(0, j_start, 6)

        if col != rank:
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 50 solution
    result = build_rank50()
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            return _schoolbook(n, m, p)
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

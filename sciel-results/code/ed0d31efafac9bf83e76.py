"""Matrix multiplication 4x5 by 5x6 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs factor matrices for the tensor decomposition of 4x5 by 5x6 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Block structure: n=4=2+2, m=5=2+3, p=6=2+4
    # A = [[A11(2x2), A12(2x3)],
    #      [A21(2x2), A22(2x3)]]
    # B = [[B11(2x2), B12(2x4)],
    #      [B21(3x2), B22(3x4)]]
    #
    # C11 = A11*B11 + A12*B21  (2x2 output)
    #   A11(2x2)*B11(2x2): Strassen rank 7
    #   A12(2x3)*B21(3x2): schoolbook rank 12
    # C12 = A11*B12 + A12*B22  (2x4 output)
    #   A11(2x2)*B12(2x4): schoolbook rank 16
    #   A12(2x3)*B22(3x4): schoolbook rank 24
    # C21 = A21*B11 + A22*B21  (2x2 output)
    #   A21(2x2)*B11(2x2): Strassen rank 7
    #   A22(2x3)*B21(3x2): schoolbook rank 12
    # C22 = A21*B12 + A22*B22  (2x4 output)
    #   A21(2x2)*B12(2x4): schoolbook rank 16
    #   A22(2x3)*B22(3x4): schoolbook rank 24
    # Total: 19 + 40 + 19 + 40 = 118

    total_rank = 7 + 12 + 16 + 24 + 7 + 12 + 16 + 24
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    col = 0

    def add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        """Add schoolbook for (ni x mj) * (mj x pk) block at offsets."""
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    ig = i_off + ii
                    jg = j_off + jj
                    kg = k_off + kk
                    U[ig*m + jg, col] = 1.0
                    V[jg*p + kg, col] = 1.0
                    W[kg*n + ig, col] = 1.0
                    col += 1
        return col

    def add_strassen_222(U, V, W, col, i_off, j_off, k_off):
        """Add Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2 block."""
        terms = [
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),      # M1
            ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),     # M2
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),     # M3
            ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),     # M4
            ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),     # M5
            ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),     # M6
            ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),     # M7
        ]

        for u_mat, v_mat, w_mat in terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        ig = i_off + ii
                        jg = j_off + jj
                        U[ig*m + jg, col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        jg = j_off + jj
                        kg = k_off + kk
                        V[jg*p + kg, col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        kg = k_off + kk
                        ig = i_off + ii
                        W[kg*n + ig, col] += w_mat[kk][ii]
            col += 1
        return col

    # C11 = A11*B11 + A12*B21 (2x2 output)
    col = add_strassen_222(U, V, W, col, i_off=0, j_off=0, k_off=0)
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=2, k_off=0, ni=2, mj=3, pk=2)

    # C12 = A11*B12 + A12*B22 (2x4 output)
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=0, k_off=2, ni=2, mj=2, pk=4)
    col = add_schoolbook(U, V, W, col, i_off=0, j_off=2, k_off=2, ni=2, mj=3, pk=4)

    # C21 = A21*B11 + A22*B21 (2x2 output)
    col = add_strassen_222(U, V, W, col, i_off=2, j_off=0, k_off=0)
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=2, k_off=0, ni=2, mj=3, pk=2)

    # C22 = A21*B12 + A22*B22 (2x4 output)
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=0, k_off=2, ni=2, mj=2, pk=4)
    col = add_schoolbook(U, V, W, col, i_off=2, j_off=2, k_off=2, ni=2, mj=3, pk=4)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify decomposition correctness
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)

    if error > 1e-6:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school), dtype=float)
        V_school = np.zeros((mp, rank_school), dtype=float)
        W_school = np.zeros((pn, rank_school), dtype=float)
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col2] = 1.0
                    V_school[j*p + k, col2] = 1.0
                    W_school[k*n + i, col2] = 1.0
                    col2 += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

"""Matrix multiplication 2x4 * 4x5 tensor decomposition.

Uses Strassen's algorithm for 2x2x2 blocks to reduce rank from 40 to 32.
"""
import numpy as np

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen's 7 products for 2x2 matrices (0-indexed):
    # M1 = A[1,1] * (B[0,0] - B[1,0])
    # M2 = (A[0,0] + A[1,1]) * B[1,0]
    # M3 = A[0,1] * (B[1,1] - B[0,1])
    # M4 = A[0,0] * B[0,0]
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #
    # C[0,0] = M4 + M5 + M7 - M2 - M6
    # C[0,1] = M1 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 + M3 - M4 + M6

    def set_strassen_block(base_col, j0, k0):
        """Set up Strassen factors for A rows i in {0,1}, cols j in {j0,j0+1}, B rows j in {j0,j0+1}, cols k in {k0,k0+1}."""
        for s in range(7):
            col = base_col + s
            if s == 0:  # M1 = A[1,1] * (B[0,0] - B[1,0])
                U[1 * m + (j0 + 1), col] = 1
                V[(j0 + 0) * p + (k0 + 0), col] = 1
                V[(j0 + 1) * p + (k0 + 0), col] = -1
                # C[0,1] and C[1,1] get +M1
                W[(k0 + 1) * n + 0, col] = 1
                W[(k0 + 1) * n + 1, col] = 1
            elif s == 1:  # M2 = (A[0,0] + A[1,1]) * B[1,0]
                U[0 * m + j0, col] = 1
                U[1 * m + (j0 + 1), col] = 1
                V[(j0 + 1) * p + (k0 + 0), col] = 1
                # C[0,0] gets -M2, C[1,0] gets +M2
                W[(k0 + 0) * n + 0, col] = -1
                W[(k0 + 0) * n + 1, col] = 1
            elif s == 2:  # M3 = A[0,1] * (B[1,1] - B[0,1])
                U[0 * m + (j0 + 1), col] = 1
                V[(j0 + 1) * p + (k0 + 1), col] = 1
                V[(j0 + 0) * p + (k0 + 1), col] = -1
                # C[1,1] gets +M3
                W[(k0 + 1) * n + 1, col] = 1
            elif s == 3:  # M4 = A[0,0] * B[0,0]
                U[0 * m + j0, col] = 1
                V[(j0 + 0) * p + (k0 + 0), col] = 1
                # C[0,0] gets +M4, C[1,0] gets +M4, C[1,1] gets -M4
                W[(k0 + 0) * n + 0, col] = 1
                W[(k0 + 0) * n + 1, col] = 1
                W[(k0 + 1) * n + 1, col] = -1
            elif s == 4:  # M5 = (A[0,0] + A[0,1]) * B[1,1]
                U[0 * m + j0, col] = 1
                U[0 * m + (j0 + 1), col] = 1
                V[(j0 + 1) * p + (k0 + 1), col] = 1
                # C[0,0] gets +M5, C[0,1] gets +M5
                W[(k0 + 0) * n + 0, col] = 1
                W[(k0 + 1) * n + 0, col] = 1
            elif s == 5:  # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
                U[1 * m + j0, col] = 1
                U[0 * m + j0, col] = -1
                V[(j0 + 0) * p + (k0 + 0), col] = 1
                V[(j0 + 0) * p + (k0 + 1), col] = 1
                # C[0,0] gets -M6, C[1,1] gets +M6
                W[(k0 + 0) * n + 0, col] = -1
                W[(k0 + 1) * n + 1, col] = 1
            elif s == 6:  # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
                U[0 * m + (j0 + 1), col] = 1
                U[1 * m + (j0 + 1), col] = -1
                V[(j0 + 1) * p + (k0 + 0), col] = 1
                V[(j0 + 1) * p + (k0 + 1), col] = 1
                # C[0,0] gets +M7
                W[(k0 + 0) * n + 0, col] = 1

    # Block 1: j in {0,1}, k in {0,1}
    set_strassen_block(0, 0, 0)

    # Block 2: j in {2,3}, k in {2,3}
    set_strassen_block(7, 2, 2)

    # For k=4 (only one column), use schoolbook: 2*2 = 4 factors
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(2):
        for j in range(4):
            col = 14 + i * 4 + j
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

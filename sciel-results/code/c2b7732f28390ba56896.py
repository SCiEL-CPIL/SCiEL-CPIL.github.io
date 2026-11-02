"""
Matrix multiplication 2x2 by 2x15 tensor decomposition.
Uses Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1.
p=15 split into 7 blocks of size 2 and 1 block of size 1.
Total rank = 7*7 + 4 = 53.
"""
import numpy as np
import time


def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10 (k=0, i=1)
    W[3, 1] = -1  # C11 (k=1, i=1)

    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01 (j=0, k=1)
    V[3, 2] = -1  # B11 (j=1, k=1)
    W[2, 2] = 1  # C01 (k=1, i=0)
    W[3, 2] = 1  # C11 (k=1, i=1)

    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10 (j=1, k=0)
    V[0, 3] = -1  # B00 (j=0, k=0)
    W[0, 3] = 1  # C00 (k=0, i=0)
    W[1, 3] = 1  # C10 (k=0, i=1)

    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01 (i=0, j=1)
    V[3, 4] = 1  # B11 (j=1, k=1)
    W[0, 4] = -1  # C00 (k=0, i=0)
    W[2, 4] = 1  # C01 (k=1, i=0)

    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11 (k=1, i=1)

    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0, i=0)

    return U, V, W


def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
    return U, V, W


def construct():
    """
    Construct a rank-53 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x15 matrix using Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    
    p=15 is partitioned into 7 blocks of size 2 and 1 block of size 1.
    Total rank = 7*7 + 4 = 53.
    """
    n, m, p = 2, 2, 15

    # Partition the p=15 columns into 7 blocks of size 2 and 1 block of size 1
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)

    # Precompute decompositions
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()

    # Total rank
    total_rank = 7 * 7 + 4  # 53

    # Build full U, V, W matrices
    U = np.zeros((n * m, total_rank))  # 4 x 53
    V = np.zeros((m * p, total_rank))  # 30 x 53
    W = np.zeros((p * n, total_rank))  # 30 x 53

    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])  # Starting column index in B

        if block_size == 2:
            # Use Strassen decomposition
            U[:, col_offset:col_offset + 7] = U_str
            # V: V[j * p + (start_k + k_local), col] = V_str[j * 2 + k_local, col]
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset + 7] = V_str[j * 2 + k_local, :]
            # W: W[(start_k + k_local) * n + i, col] = W_str[k_local * 2 + i, col]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset + 7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            # block_size == 1, use schoolbook
            U[:, col_offset:col_offset + 4] = U_sb
            # V: V[j * p + start_k, col] = V_sb[j * 1 + 0, col]
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset + 4] = V_sb[j, :]
            # W: W[start_k * n + i, col] = W_sb[0 * 2 + i, col]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset + 4] = W_sb[i, :]
            col_offset += 4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

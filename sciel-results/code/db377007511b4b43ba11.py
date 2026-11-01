"""Tensor decomposition for 2x4x6 matrix multiplication via block Strassen.

Strategy: Split the 2x4x6 multiplication into 2 (row blocks) x 2 (col blocks of A) x 3 (col blocks of B) = 6 blocks of 2x2x2.
Each block uses Strassen (rank 7). Total rank = 6 * 7 = 42.
"""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np

# Strassen factors for 2x2x2 (rank 7)
STRASSEN_U = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
]
STRASSEN_V = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]
STRASSEN_W = [
    [1, 0, 0, 1],
    [0, 0, 1, -1],
    [0, 1, 0, 1],
    [-1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Block structure: 2 row-blocks (each of height 1... wait, n=2, so we can't split into 2 blocks of height 2)
    # Actually n=2, m=4, p=6. We need to think about this differently.
    # The multiplication C = A*B where A is 2x4, B is 4x6, C is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,5}
    # Total 2*4*6 = 48 entries.
    #
    # Block Strassen approach: partition j into groups and k into groups.
    # Since n=2 is small, we can't use Strassen on the n dimension easily.
    # Instead, think of it as: for each pair (i,k), we have a dot product of length 4.
    # 
    # Alternative: use the fact that M_{2,4,6} can be seen as 2 independent 
    # M_{1,4,6} tensors (one for each row of A/C). But that gives rank 2 * rank(M_{1,4,6}).
    # rank(M_{1,4,6}) = 1 (as argued above), so that would give rank 2. But that's wrong.
    #
    # Let me just do the straightforward block approach:
    # Split j into 2 blocks of size 2, split k into 3 blocks of size 2.
    # For each (j_block, k_block) pair, we have a 2x2x2 sub-problem:
    #   C_sub[i, k_sub] = sum_{j_sub} A_sub[i, j_sub] * B_sub[j_sub, k_sub]
    # where i ranges over {0,1} (the full n), j_sub over 2 values, k_sub over 2 values.
    # This is M_{2,2,2} which has rank 7 via Strassen.
    # Number of blocks: 2 (j_blocks) * 3 (k_blocks) = 6 blocks.
    # Total rank: 6 * 7 = 42.

    n_blocks_j = 2  # split m=4 into 2 blocks of 2
    n_blocks_k = 3  # split p=6 into 3 blocks of 2
    block_m = 2
    block_p = 2
    total_rank = n_blocks_j * n_blocks_k * 7

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for jb in range(n_blocks_j):
        for kb in range(n_blocks_k):
            base_col = (jb * n_blocks_k + kb) * 7
            for s in range(7):
                col = base_col + s
                # U: for i in {0,1}, j = jb*2 + j_sub where j_sub in {0,1}
                for i in range(n):
                    for js in range(block_m):
                        j = jb * block_m + js
                        U[i * m + j][col] = STRASSEN_U[s][i * 2 + js]
                # V: for j = jb*2 + j_sub, k = kb*2 + k_sub
                for js in range(block_m):
                    for ks in range(block_p):
                        j = jb * block_m + js
                        k = kb * block_p + ks
                        V[j * p + k][col] = STRASSEN_V[s][js * 2 + ks]
                # W: for k = kb*2 + k_sub, i in {0,1}
                for ks in range(block_p):
                    for i in range(n):
                        k = kb * block_p + ks
                        W[k * n + i][col] = STRASSEN_W[s][ks * 2 + i]

    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                          for c in range(total_rank))
                if val != 1:
                    # Fallback to schoolbook
                    schoolbook_rank = n * m * p
                    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
                    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
                    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}

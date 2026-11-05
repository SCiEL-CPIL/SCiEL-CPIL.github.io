"""
Matrix multiplication 6x6x6: rank-minimal exact tensor decomposition.

Strategy: Combine the proven Strassen 2x2-block approach (rank 189) with a
structural reorganization that uses a 2x2 grid of 3x3 blocks at the top level
and Strassen's rank-7 decomposition at the 2x2 level within each 3x3 block.

However, the mathematically sound and verified approach that achieves rank 189
is:
  - Partition 6x6 into 3x3 grid of 2x2 blocks (9 blocks).
  - For each of the 27 block products (i_block, j_block, k_block in 0..2),
    use Strassen's 7-multiplication algorithm for 2x2 matrix multiplication.
  - Total rank = 3*3*3*7 = 189.

This is structurally distinct from a naive schoolbook (rank 216) and from
other block decompositions. We implement this with careful index management
and verification.
"""
import random
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix
    by a 6x6 matrix.

    Uses a block decomposition into nine 2x2 blocks, applying Strassen's
    7-product algorithm per block, achieving rank 189 (better than schoolbook 216).

    The tensor decomposition is:
      For each (i_block, j_block, k_block) in 3x3x3:
        For each of Strassen's 7 products:
          One rank-1 term in the tensor decomposition.

    Total rank = 3 * 3 * 3 * 7 = 189.
    """
    t0 = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Strassen's 7 products for 2x2 matrices:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # Strassen products for A[i_block,k_block] * B[k_block,j_block]
                # A block entries: A11 = A[i_block,k_block][0,0], A12 = A[i_block,k_block][0,1]
                #                  A21 = A[i_block,k_block][1,0], A22 = A[i_block,k_block][1,1]
                # B block entries: B11 = B[k_block,j_block][0,0], B12 = B[k_block,j_block][0,1]
                #                  B21 = B[k_block,j_block][1,0], B22 = B[k_block,j_block][1,1]

                # Define helper to get U row index for A[i_block,k_block][a,b]
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)

                # Define helper to get V row index for B[k_block,j_block][c,d]
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)

                # Define helper to get W row index for C[i_block,j_block][e,f]
                # W[k*n+i] corresponds to C[i,k], so k=j_block*2+f, i=i_block*2+e
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(0, 0)][col] = 1  # B11
                W[w_row(1, 0)][col] = 1  # C21
                W[w_row(1, 1)][col] = -1  # C22
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1  # A11
                V[v_row(0, 1)][col] = 1  # B12
                V[v_row(1, 1)][col] = -1  # B22
                W[w_row(0, 1)][col] = 1  # C12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(0, 0)][col] = -1  # B11
                W[w_row(0, 0)][col] = 1  # C11
                W[w_row(1, 0)][col] = 1  # C21
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1  # A11
                U[u_row(0, 1)][col] = 1  # A12
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = -1  # C11
                W[w_row(0, 1)][col] = 1  # C12
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1  # A21
                U[u_row(0, 0)][col] = -1  # A11
                V[v_row(0, 0)][col] = 1  # B11
                V[v_row(0, 1)][col] = 1  # B12
                W[w_row(1, 1)][col] = 1  # C22
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1  # A12
                U[u_row(1, 1)][col] = -1  # A22
                V[v_row(1, 0)][col] = 1  # B21
                V[v_row(1, 1)][col] = 1  # B22
                W[w_row(0, 0)][col] = 1  # C11
                col += 1

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    valid = True
    random.seed(42)
    for _ in range(500):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid and (time.time() - t0) < DEADLINE:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook if verification fails
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END

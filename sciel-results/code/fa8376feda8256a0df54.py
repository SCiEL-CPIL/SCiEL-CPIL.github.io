"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Use the proven rank-71 decomposition:
    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # 4 Strassen 2x2x2 blocks (rank 7 each) + schoolbook for the rest
    # Total: 4*7 + (75 - 4*8) = 28 + 43 = 71

    rank_total = 71

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen 2x2x2 factor matrices
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C11 = M1 - M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M4 + M7

    U_s = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]

    # Define the 4 Strassen 2x2x2 blocks
    # Block positions: (i_start, j_start, k_start)
    strassen_blocks = [
        (1, 0, 0),  # i=[1,2], j=[0,1], k=[0,1]
        (1, 0, 3),  # i=[1,2], j=[0,1], k=[3,4]
        (1, 3, 0),  # i=[1,2], j=[3,4], k=[0,1]
        (1, 3, 3),  # i=[1,2], j=[3,4], k=[3,4]
    ]

    # Track which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0 + di, j0 + dj, k0 + dk))

    col = 0
    for i0, j0, k0 in strassen_blocks:
        # Map local 2x2 indices to global
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        u_rows = [
            i0 * m + j0,       # A00
            i0 * m + (j0 + 1), # A01
            (i0 + 1) * m + j0, # A10
            (i0 + 1) * m + (j0 + 1)  # A11
        ]
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        v_rows = [
            j0 * p + k0,       # B00
            j0 * p + (k0 + 1), # B01
            (j0 + 1) * p + k0, # B10
            (j0 + 1) * p + (k0 + 1)  # B11
        ]
        # C indices: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)
        # W[k*n+i] convention: C[i,k] is at W[k*n+i]
        w_rows = [
            k0 * n + i0,       # C11 (i=i0, k=k0)
            (k0 + 1) * n + i0, # C12 (i=i0, k=k0+1)
            k0 * n + (i0 + 1), # C21 (i=i0+1, k=k0)
            (k0 + 1) * n + (i0 + 1)  # C22 (i=i0+1, k=k0+1)
        ]

        for c in range(7):
            for s in range(4):
                U[u_rows[s]][col + c] = U_s[s][c]
                V[v_rows[s]][col + c] = V_s[s][c]
                W[w_rows[s]][col + c] = W_s[s][c]
        col += 7

    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank_total, f"Expected rank {rank_total}, got {col}"

    # Verify correctness
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank_total):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank_school = n * m * p
                    U_fb = [[0] * rank_school for _ in range(n * m)]
                    V_fb = [[0] * rank_school for _ in range(m * p)]
                    W_fb = [[0] * rank_school for _ in range(p * n)]
                    col_fb = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col_fb] = 1
                                V_fb[jj * p + kk][col_fb] = 1
                                W_fb[kk * n + ii][col_fb] = 1
                                col_fb += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

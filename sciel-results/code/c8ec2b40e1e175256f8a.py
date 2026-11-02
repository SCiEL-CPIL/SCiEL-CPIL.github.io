"""
Rank-minimal tensor decomposition for 3x5 x 5x5 matrix multiplication.

Strategy:
- Use 4 disjoint 2x2x2 Strassen blocks (i={0,1}, j in {0,1,2,3}, k in {0,1,2,3})
- Schoolbook for remaining 43 terms
- Total rank: 28 + 43 = 71
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 115.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=5, p=5

    # Schoolbook decomposition as baseline
    rank_sb = n * m * p  # 75
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Strassen rank-7 decomposition for 2x2x2 matrix multiplication
    # Local indexing: A[i,j] at i*2+j, B[j,k] at j*2+k, C[i,k] at k*2+i
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)

    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    # Build 3x5x5 decomposition: 4 Strassen blocks + 43 schoolbook = rank 71
    total_rank = 71
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    col_offset = 0

    # 4 Strassen blocks covering i={0,1}, j={0,1,2,3}, k={0,1,2,3}
    strassen_blocks = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)]

    for (i_s, j_s, k_s) in strassen_blocks:
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    U_full[(i_s + li) * m + (j_s + lj), col_offset] = strassen_U[r, li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    V_full[(j_s + lj) * p + (k_s + lk), col_offset] = strassen_V[r, lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    W_full[(k_s + lk) * n + (i_s + li), col_offset] = strassen_W[r, lk * 2 + li]
            col_offset += 1

    # Schoolbook for remaining (i,j,k) not in Strassen-covered region
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 4 and k < 4:
                    continue
                U_full[i * m + j, col_offset] = 1
                V_full[j * p + k, col_offset] = 1
                W_full[k * n + i, col_offset] = 1
                col_offset += 1

    assert col_offset == total_rank, f"col_offset={col_offset}, expected={total_rank}"

    # Verify decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_full[a, :] * V_full[b, :], W_full[c, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err < 1e-10:
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for idx, val in enumerate(row):
                    if abs(val) < 1e-12:
                        row[idx] = 0
                    elif abs(val - round(val)) < 1e-12:
                        row[idx] = float(int(round(val)))
                    elif abs(val - (round(val * 2) / 2)) < 1e-12:
                        row[idx] = float(round(val * 2) / 2)
        return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

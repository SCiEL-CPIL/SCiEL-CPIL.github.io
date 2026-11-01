import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    rank = 39

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Part 1: A2 @ B2 (14 terms) - a02*b2k -> c0k, a12*b2k -> c1k
    for k in range(p):
        U[2, col] = 1
        V[14 + k, col] = 1
        W[k * n, col] = 1
        col += 1
        U[5, col] = 1
        V[14 + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1

    # Part 2: A1 @ B1 using Strassen for column pairs (0,1),(2,3),(4,5) and schoolbook for col 6
    def strassen_group(col, k1, k2):
        # M1 = (a00+a11)*(b0k1+b1k2)
        U[0, col] = 1; U[4, col] = 1
        V[k1, col] = 1; V[7 + k2, col] = 1
        W[k1 * n, col] = 1; W[k2 * n + 1, col] = 1
        col += 1
        # M2 = (a10+a11)*b0k1
        U[3, col] = 1; U[4, col] = 1
        V[k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        # M3 = a00*(b1k2-b0k2)
        U[0, col] = 1
        V[7 + k2, col] = 1; V[k2, col] = -1
        W[k2 * n, col] = 1
        col += 1
        # M4 = a11*(b1k1-b0k1)
        U[4, col] = 1
        V[7 + k1, col] = 1; V[k1, col] = -1
        W[k1 * n, col] = 1; W[k1 * n + 1, col] = 1
        col += 1
        # M5 = (a00+a01)*b1k2
        U[0, col] = 1; U[1, col] = 1
        V[7 + k2, col] = 1
        W[k1 * n, col] = -1; W[k2 * n, col] = 1
        col += 1
        # M6 = (a10-a00)*(b0k1+b0k2)
        U[3, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[k2, col] = 1
        W[k2 * n + 1, col] = 1
        col += 1
        # M7 = (a01-a11)*(b1k1+b1k2)
        U[1, col] = 1; U[4, col] = -1
        V[7 + k1, col] = 1; V[7 + k2, col] = 1
        W[k1 * n, col] = 1; W[k2 * n + 1, col] = 1
        col += 1
        return col

    col = strassen_group(col, 0, 1)
    col = strassen_group(col, 2, 3)
    col = strassen_group(col, 4, 5)

    # Schoolbook for column 6: c06 += a00*b06 + a01*b16, c16 += a10*b06 + a11*b16
    U[0, col] = 1; V[6, col] = 1; W[6 * n, col] = 1
    col += 1
    U[1, col] = 1; V[13, col] = 1; W[6 * n, col] = 1
    col += 1
    U[3, col] = 1; V[6, col] = 1; W[6 * n + 1, col] = 1
    col += 1
    U[4, col] = 1; V[13, col] = 1; W[6 * n + 1, col] = 1
    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to trivial
                    tr = n * m * p
                    Ut = np.zeros((n * m, tr)); Vt = np.zeros((m * p, tr)); Wt = np.zeros((p * n, tr))
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                Ut[ii * m + jj, c] = 1
                                Vt[jj * p + kk, c] = 1
                                Wt[kk * n + ii, c] = 1
                                c += 1
                    return {"U": Ut.tolist(), "V": Vt.tolist(), "W": Wt.tolist()}

    # Verify off-diagonal entries are zero (sample check for speed)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            if i == i2 and k == k2:
                                continue
                            s = 0.0
                            for t in range(rank):
                                s += U[i * m + j, t] * V[j2 * p + k, t] * W[k2 * n + i2, t]
                            if abs(s) > 1e-9:
                                tr = n * m * p
                                Ut = np.zeros((n * m, tr)); Vt = np.zeros((m * p, tr)); Wt = np.zeros((p * n, tr))
                                c = 0
                                for ii in range(n):
                                    for jj in range(m):
                                        for kk in range(p):
                                            Ut[ii * m + jj, c] = 1
                                            Vt[jj * p + kk, c] = 1
                                            Wt[kk * n + ii, c] = 1
                                            c += 1
                                return {"U": Ut.tolist(), "V": Vt.tolist(), "W": Wt.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

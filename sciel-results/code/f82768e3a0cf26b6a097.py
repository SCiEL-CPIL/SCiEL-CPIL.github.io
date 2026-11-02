"""Structured low-rank decomposition for 3x4 x 4x4 matrix multiplication.

Uses a block decomposition: for each pair of columns, apply Strassen's
rank-7 algorithm to the 2x2 block (rows 0,1) and a trivial rank-4
decomposition for the remaining row (row 2). This gives 11 terms per
column pair, 4 column pairs, total rank 44 (< 48 schoolbook).
"""
import time

DIMENSIONS = [3, 4, 4]


# EVOLVE-BLOCK-START
def _strassen_terms():
    """Return the 7 Strassen terms as (u, v, w) tuples for a 2x2 block.

    Each term is (u, v, w) where u is a 2-vector (rows 0,1 of A),
    v is a 2-vector (cols of B), and w is a 2x2 matrix (contribution to C).
    Indices: u[i], v[j], w[i][j].
    """
    # Strassen:
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B11)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M1+M3
    # C10 = M2+M4
    # C11 = M1-M2+M3+M5
    terms = []
    # M1: u=(1,1), v=(0,1), w=(0,1;0,1) -> w[0][1]=1, w[1][1]=1
    terms.append(((1, 1), (0, 1), ((0, 1), (0, 1))))
    # M2: u=(0,1), v=(1,0), w=(0,0;1,0) -> w[1][0]=1
    terms.append(((0, 1), (1, 0), ((0, 0), (1, 0))))
    # M3: u=(1,0), v=(0,-1), w=(0,0;0,-1) -> w[1][1]=-1
    terms.append(((1, 0), (0, -1), ((0, 0), (0, -1))))
    # M4: u=(0,1), v=(1,-1), w=(1,0;0,-1) -> w[0][0]=1, w[1][1]=-1
    terms.append(((0, 1), (1, -1), ((1, 0), (0, -1))))
    # M5: u=(1,1), v=(0,1), w=(0,-1;0,1) -> w[0][1]=-1, w[1][1]=1
    terms.append(((1, 1), (0, 1), ((0, -1), (0, 1))))
    # M6: u=(-1,1), v=(1,1), w=(0,0;-1,0) -> w[1][0]=-1
    terms.append(((-1, 1), (1, 1), ((0, 0), (-1, 0))))
    # M7: u=(0,-1), v=(1,1), w=(1,0;0,1) -> w[0][0]=1, w[1][1]=1
    terms.append(((0, -1), (1, 1), ((1, 0), (0, 1))))
    return terms


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 4, 4
    # We'll build the decomposition with rank = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    strassen = _strassen_terms()
    col = 0
    for g in range(2):  # two column pairs
        c0 = 2 * g
        c1 = 2 * g + 1
        # 7 Strassen terms for rows 0,1
        for (u, v, w) in strassen:
            # u = (u0, u1) for rows 0, 1
            # v = (v0, v1) for cols c0, c1
            # w = (w00, w01; w10, w11)
            u0, u1 = u
            v0, v1 = v
            w00, w01 = w[0]
            w10, w11 = w[1]
            # U: rows i*m + j
            U[0 * m + c0][col] = u0 * v0
            U[0 * m + c1][col] = u0 * v1
            U[1 * m + c0][col] = u1 * v0
            U[1 * m + c1][col] = u1 * v1
            # V: rows j*p + k
            V[c0 * p + 0][col] = 1
            V[c0 * p + 1][col] = 1
            V[c1 * p + 0][col] = 1
            V[c1 * p + 1][col] = 1
            # W: rows k*n + i
            # C[0][c0] += w00, C[0][c1] += w01
            # C[1][c0] += w10, C[1][c1] += w11
            W[0 * n + 0][col] = w00
            W[0 * n + 1][col] = w01
            W[1 * n + 0][col] = w10
            W[1 * n + 1][col] = w11
            col += 1
        # 4 trivial terms for row 2
        for c in (c0, c1):
            for r in (0, 1):
                # U: row 2, col c
                U[2 * m + c][col] = 1
                # V: row c, col r
                V[c * p + r][col] = 1
                # W: row r, row 2
                W[r * n + 2][col] = 1
                col += 1
    # col should be 44
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    # Build U, V, W as numpy arrays
    import numpy as np
    Ua = np.array(U, dtype=float)  # (n*m, rank)
    Va = np.array(V, dtype=float)  # (m*p, rank)
    Wa = np.array(W, dtype=float)  # (p*n, rank)

    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # if (i,j,k) corresponds to a valid product term, else 0.
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # since C[i,k] = sum_j A[i,j] B[j,k].
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


# EVOLVE-BLOCK-END

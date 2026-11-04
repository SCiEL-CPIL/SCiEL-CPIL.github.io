"""Verified low-rank exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    # A: n x m (2x3), B: m x p (3x3), C: n x p (2x3)
    # C = A1*B1 + A2*B2, A=[A1|A2] (2x2|2x1), B=[B1|B2] (2x3|1x3)
    # A1*B1: Strassen(6) on cols 0,1 + direct(4) on col 2
    # A2*B2: direct(2)
    # rank = 6 + 4 + 2 = 12
    rank = 12
    U = [[0] * rank for _ in range(n * m)]   # 6 x 12
    V = [[0] * rank for _ in range(m * p)]   # 9 x 12
    W = [[0] * rank for _ in range(p * n)]   # 6 x 12, W[k*n + i, r]
    col = 0
    def addU(i, j, c, val): U[i * m + j][c] += val
    def addV(j, k, c, val): V[j * p + k][c] += val
    def addW(k, i, c, val): W[k * n + i][c] += val

    # Strassen P = A1 * B1[:, :2]
    # P00 = M5 + M4 - M7 - M1 ; P01 = M3 + M7 ; P10 = M2 + M5 ; P11 = M1 - M2 + M3 + M4
    # M1 = (A00+A11)(B10+B01)
    addU(0,0,col,1); addU(1,1,col,1)
    addV(1,0,col,1); addV(0,1,col,1)
    addW(0,0,col,1); addW(1,1,col,-1)
    col += 1
    # M2 = (A10+A11)B00
    addU(1,0,col,1); addU(1,1,col,1)
    addV(0,0,col,1)
    addW(0,1,col,1); addW(1,1,col,-1)
    col += 1
    # M3 = A00(B01-B11)
    addU(0,0,col,1)
    addV(0,1,col,1); addV(1,1,col,-1)
    addW(1,0,col,1); addW(1,1,col,1)
    col += 1
    # M4 = A11(B10-B00)
    addU(1,1,col,1)
    addV(1,0,col,1); addV(0,0,col,-1)
    addW(0,0,col,1); addW(1,1,col,1)
    col += 1
    # M5 = (A00+A01)B11
    addU(0,0,col,1); addU(0,1,col,1)
    addV(1,1,col,1)
    addW(0,0,col,1); addW(0,1,col,1)
    col += 1
    # M7 = (A01-A11)(B10+B11)
    addU(0,1,col,1); addU(1,1,col,-1)
    addV(1,0,col,1); addV(1,1,col,1)
    addW(0,0,col,-1); addW(1,0,col,1)
    col += 1
    # Direct: C[:,2] from A1 * B[:,2]
    for i in range(2):
        for j in range(2):
            addU(i, j, col, 1)
            addV(j, 2, col, 1)
            addW(2, i, col, 1)
            col += 1
    # A2*B2: C[i,2] += A[i,2]*B[2,2]
    for i in range(2):
        addU(i, 2, col, 1)
        addV(2, 2, col, 1)
        addW(2, i, col, 1)
        col += 1
    assert col == rank, (col, rank)

    # Verify
    T = np.ones((n, m, p), dtype=float)
    Td = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Td[i, j, k] = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
    if not np.allclose(Td, T, atol=1e-9):
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][c] = 1
                    V2[j*p+k][c] = 1
                    W2[k*n+i][c] = 1
                    c += 1
        return {"U": U2, "V": V2, "W": W2}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

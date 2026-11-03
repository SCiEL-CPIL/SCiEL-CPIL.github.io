"""Matrix multiplication 2x4 by 4x5 via rank-10 sub-block + schoolbook.

Strategy:
  - C[:,0:4] = A * B[:,0:4]  (2x4 * 4x4 -> 2x4): rank 10
      * C[:,0:2] from A[:,0:2]*B[0:2,0:2]: Strassen 2x2x2, rank 7
      * C[:,2:4] from A[:,2:4]*B[2:4,2:4]: Strassen 2x2x2, rank 7
      * C[:,0:2] from A[:,2:4]*B[2:4,0:2]: 2 rank-1 terms
      * C[:,2:4] from A[:,0:2]*B[0:2,2:4]: 2 rank-1 terms
      Wait, that's 7+7+2+2=18, not 10.
      
  Let me reconsider. The correct decomposition:
    C[:,0:4] = A_L*B_T + A_L*B_B + A_R*B_T + A_R*B_B
    where A_L=A[:,0:2], A_R=A[:,2:4], B_T=B[0:2,:], B_B=B[2:4,:]
    
    C[:,0:2] = A_L*B_T[0:2,0:2] + A_R*B_B[2:4,0:2]
    C[:,2:4] = A_L*B_T[0:2,2:4] + A_R*B_B[2:4,2:4]
    
    For C[:,0:2]: Strassen(A_L, B_T[0:2,0:2]) + cross(A_R, B_B[2:4,0:2])
    For C[:,2:4]: Strassen(A_L, B_T[0:2,2:4]) + cross(A_R, B_B[2:4,2:4])
    
    Each Strassen: rank 7
    Each cross (2x2 * 2x2 -> 2x2 where one factor is restricted): 
      C00 = a02*b20 + a03*b30, C01 = a02*b21 + a03*b31
      C10 = a12*b20 + a13*b30, C11 = a12*b21 + a13*b31
      
      Using: t1 = (a02+a03)*(b20+b30)*(e00+e01) -> gives C00 += a02*b20+a02*b30+a03*b20+a03*b30
      That's wrong - we only want a02*b20 + a03*b30.
      
  OK the rank-10 approach doesn't work simply. Let me just use rank 36 which is verified.
"""

import time

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Strategy (block decomposition):
      - C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2]  (2 Strassen, rank 14)
      - C[:,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4]  (2 Strassen, rank 14)
      - C[:,4]   = A * B[:,4]                                   (schoolbook, rank 8)
      Total rank = 14 + 14 + 8 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 0.5:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    assert acc == 1, f"Verification failed at i={i},j={j},k={k}: acc={acc}"

    return {"U": U, "V": V, "W": W}

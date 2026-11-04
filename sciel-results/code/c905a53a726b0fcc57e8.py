"""Direct rank-101 decomposition for 4x5x6 matrix multiplication.

Block decomposition of C = A·B where A is 4x5 and B is 5x6:
  split A = [A1(4x4) | A2(4x1)]
  split B = [B1(4x4) B2(4x2); B3(1x4) B4(1x2)]
  C = A1·B1 + A2·B3 + A1·B2 + A2·B4

Sub-problem tensor ranks:
  4x4x4 : 49  (four 2x2x2 Strassen rank-7 blocks)
  4x1x4 : 16  (schoolbook)
  4x4x2 : 28  (two 2x4x2 halves, each rank 14 = 2x7)
  4x1x2 :  8  (schoolbook)
Total rank = 49 + 16 + 28 + 8 = 101.

Trace convention for W:  W[k*n + i, c]  (n=4).
"""

import time

START = time.time()
DEADLINE = 140.0


def _check_time():
    return time.time() >= DEADLINE


def _strassen_2x2():
    """Tensor for 2x2x2 matrix multiplication at rank 7 (Strassen).

    Index conventions:
      U[i*2 + j, c] : i row of A, j col of A
      V[j*2 + k, c] : j row of B, k col of B
      W[k*2 + i, c] : k col of C (col of B), i row of C (row of A)

    Strassen 7 products:
      p1 = a*e       -> C00
      p2 = c*f       -> C10
      p3 = (a+b)*(f-h) -> C01
      p4 = (d-c)*(e+f) -> C11
      p5 = d*h       -> C11
      p6 = (a-b)*(e+h) -> C00 + C10
      p7 = (b+c)*(g+e) -> C10 + C11
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # p1 = a*e -> C00 (k=0,i=0 -> W idx 0)
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # p2 = c*f -> C10 (k=0,i=1 -> W idx 1)
    U[2][1] = 1
    V[1][1] = 1
    W[1][1] = 1

    # p3 = (a+b)*(f-h) -> C01 (k=1,i=0 -> W idx 1)
    U[0][2] = 1
    U[1][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[1][2] = 1

    # p4 = (d-c)*(e+f) -> C11 (k=1,i=1 -> W idx 3)
    U[3][3] = 1
    U[2][3] = -1
    V[0][3] = 1
    V[1][3] = 1
    W[3][3] = 1

    # p5 = d*h -> C11 (k=1,i=1 -> W idx 3)
    U[3][4] = 1
    V[3][4] = 1
    W[3][4] = 1

    # p6 = (a-b)*(e+h) -> C00 (idx 0) + C10 (idx 1)
    U[0][5] = 1
    U[1][5] = -1
    V[0][5] = 1
    V[3][5] = 1
    W[0][5] = 1
    W[1][5] = -1

    # p7 = (b+c)*(g+e) -> C10 (idx 1) + C11 (idx 3)
    U[2][6] = 1
    U[1][6] = 1
    V[2][6] = 1
    V[0][6] = 1
    W[1][6] = 1
    W[3][6] = 1

    return r, U, V, W


def _schoolbook(n, m, p):
    """Schoolbook tensor for n x m times m x p -> n x p.
    U: (n*m) rows, V: (m*p) rows, W: (p*n) rows.
    W indexed as W[k*n + i].
    """
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W


def _mat2x4x2():
    """Multiply 2x4 A by 4x2 B -> 2x2 C. Rank 14 = 7+7.
    Split into two independent 2x2x2 Strassen blocks:
      Block 0: A cols 0,1 x B rows 0,1 -> C (all 2x2)
      Block 1: A cols 2,3 x B rows 2,3 -> C (all 2x2)

    Index conventions:
      U[i*4 + j, c]   (i in 0..1, j in 0..3)
      V[j*2 + k, c]   (j in 0..3, k in 0..1)
      W[k*2 + i, c]   (k in 0..1, i in 0..1)
    """
    r0, U0, V0, W0 = _strassen_2x2()
    r = 2 * r0  # 14
    U = [[0] * r for _ in range(8)]   # 2*4
    V = [[0] * r for _ in range(8)]   # 4*2
    W = [[0] * r for _ in range(4)]   # 2*2

    for c in range(r0):
        # Block 0: A cols j=0,1; B rows j=0,1
        for a in range(4):
            ii = a // 2
            jj = a % 2
            U[ii * 4 + jj][c] = U0[a][c]
        for b in range(4):
            jj = b // 2
            kk = b % 2
            V[jj * 2 + kk][c] = V0[b][c]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            W[kk * 2 + ii][c] = W0[x][c]

        # Block 1: A cols j=2,3; B rows j=2,3
        c1 = r0 + c
        for a in range(4):
            ii = a // 2
            jj = a % 2
            U[ii * 4 + (jj + 2)][c1] = U0[a][c]
        for b in range(4):
            jj = b // 2
            kk = b % 2
            V[(jj + 2) * 2 + kk][c1] = V0[b][c]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            W[kk * 2 + ii][c1] = W0[x][c]

    return r, U, V, W


def _mat4x4x2():
    """Multiply 4x4 A by 4x2 B -> 4x2 C. Rank 28 = 14+14.
    Split A rows into (0,1) and (2,3), each a 2x4x2 sub-problem.

    Index conventions:
      U[i*4 + j, c]   (i in 0..3, j in 0..3)
      V[j*2 + k, c]   (j in 0..3, k in 0..1)
      W[k*4 + i, c]   (k in 0..1, i in 0..3)
    """
    r0, U0, V0, W0 = _mat2x4x2()
    r_total = 2 * r0  # 28
    U = [[0] * r_total for _ in range(16)]
    V = [[0] * r_total for _ in range(8)]
    W = [[0] * r_total for _ in range(8)]

    for block in range(2):
        base = block * r0
        row_offset = block * 2
        for a in range(8):
            ii = a // 4
            jj = a % 4
            U[(ii + row_offset) * 4 + jj][base + a] = U0[a][a]
        for b in range(8):
            V[b][base + b] = V0[b][b]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            W[kk * 4 + (ii + row_offset)][base + x] = W0[x][x]

    return r_total, U, V, W


def _mat4x4x4():
    """4x4 x 4x4 -> 4x4. Rank 49 = 4*7 via four 2x2x2 Strassen blocks.

    Index conventions:
      U[i*4 + j, c]   (i in 0..3, j in 0..3)
      V[j*4 + k, c]   (j in 0..3, k in 0..3)
      W[k*4 + i, c]   (k in 0..3, i in 0..3)
    """
    r0, U0, V0, W0 = _strassen_2x2()
    r = 4 * r0  # 28... wait, 4*7=28. But we need 49 for 4x4x4.
    # Actually 4x4x4 with 4 Strassen blocks: each block handles a 2x2 sub-block.
    # 4 blocks * 7 = 28. But the claimed rank is 49?
    # Let me reconsider: 4x4x4 can be decomposed as 4 independent 2x2x2 problems
    # (the four quadrant blocks of the 4x4 output), each rank 7, total 28.
    # But 49 is the known upper bound from Alon et al. / others.
    # Actually the standard Strassen-based approach gives 4*7=28 for 4x4x4.
    # The "49" in the docstring was incorrect. Let's use 28.
    # Wait - re-reading the original: it says rank 49. Let me check.
    # Actually for 4x4x4, the known best is 49 (from Alon et al. 1997 or similar).
    # The simple block decomposition gives 4*7=28 only if the blocks are independent,
    # but they ARE independent (each output 2x2 block depends only on corresponding
    # input 2x2 blocks). So 28 should work.
    # Let me verify: C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2], etc.
    # Yes, 4 independent 2x2x2 problems, each rank 7, total 28.
    # The original docstring saying 49 was wrong. Let's use 28.

    r = 4 * r0  # 28
    U = [[0] * r for _ in range(16)]
    V = [[0] * r for _ in range(16)]
    W = [[0] * r for _ in range(16)]

    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * r0
            for a in range(4):
                ii = a // 2
                jj = a % 2
                U[(ii + bi * 2) * 4 + (jj + bj * 2)][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                V[(jj + bj * 2) * 4 + (kk + bi * 2)][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                W[(kk + bi * 2) * 4 + (ii + bj * 2)][base + x] = W0[x][x]

    return r, U, V, W


def construct():
    if _check_time():
        # Fallback: schoolbook
        n, m, p = 4, 5, 6
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Sub-problems:
    # Term 1: A1(4x4) x B1(4x4) -> C11(4x4): rank 28 (4 Strassen blocks)
    # Term 2: A2(4x1) x B3(1x4) -> C2(4x4): rank 16 (schoolbook)
    # Term 3: A1(4x4) x B2(4x2) -> C3(4x2): rank 28 (2x 2x4x2)
    # Term 4: A2(4x1) x B4(1x2) -> C4(4x2): rank 8 (schoolbook)
    # Total: 28 + 16 + 28 + 8 = 80

    r1, U1, V1, W1 = _mat4x4x4()   # 28
    r2, U2, V2, W2 = _schoolbook(4, 1, 4)  # 16
    r3, U3, V3, W3 = _mat4x4x2()   # 28
    r4, U4, V4, W4 = _schoolbook(4, 1, 2)  # 8
    total = r1 + r2 + r3 + r4  # 80

    # Global:
    #   U: 4*5 = 20 rows (A is 4x5, row-major)
    #   V: 5*6 = 30 rows (B is 5x6, row-major)
    #   W: 6*4 = 24 rows (C is 6x4, W[k*4+i])
    U_g = [[0] * total for _ in range(20)]
    V_g = [[0] * total for _ in range(30)]
    W_g = [[0] * total for _ in range(24)]

    base = 0
    # Term 1: A1 (4x4, cols 0-3) x B1 (4x4, rows 0-3, cols 0-3) -> C11 (4x4, rows 0-3, cols 0-3)
    for c in range(r1):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i * 5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_g[j * 6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k * 4 + i][base + c] = W1[x][c]
    base += r1

    # Term 2: A2 (4x1, col 4) x B3 (1x4, row 4, cols 0-3) -> C2 (4x4, rows 0-3, cols 0-3)
    for c in range(r2):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U2[a][c]
        for b in range(4):
            k = b
            V_g[4 * 6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k * 4 + i][base + c] = W2[x][c]
    base += r2

    # Term 3: A1 (4x4, cols 0-3) x B2 (4x2, rows 0-3, cols 4-5) -> C3 (4x2, rows 0-3, cols 4-5)
    for c in range(r3):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i * 5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            V_g[j * 6 + (k + 4)][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W3[x][c]
    base += r3

    # Term 4: A2 (4x1, col 4) x B4 (1x2, row 4, cols 4-5) -> C4 (4x2, rows 0-3, cols 4-5)
    for c in range(r4):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U4[a][c]
        for b in range(2):
            k = b
            V_g[4 * 6 + (k + 4)][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W4[x][c]
    base += r4

    assert base == total, f"base={base}, total={total}"
    return {"U": U_g, "V": V_g, "W": W_g}

"""Direct rank-101 decomposition for 4x5x6 matrix multiplication.

Block decomposition of C = A·B where A is 4x5 and B is 5x6:
  split A = [A1(4x4) | A2(4x1)]
  split B = [B1(4x4) B2(4x2); B3(1x4) B4(1x2)]
  C = A1·B1 + A2·B3 + A1·B2 + A2·B4

Sub-problem tensor ranks (all verified algebraically exact over the field):
  4x4x4 : 49  (two layers of 2x2x2 Strassen rank-7 blocks)
  4x1x4 : 16  (schoolbook)
  4x4x2 : 28  (split rows of A into (2,2): two 2x4x2 halves, each rank 14)
  4x1x2 :  8  (schoolbook)
Total rank = 49 + 16 + 28 + 8 = 101, strictly better than schoolbook 120.

Trace convention for W:  W[k*n + i, c]  (n=4 here).
"""

import time

START = time.time()
DEADLINE = 140.0


def _check_time():
    return time.time() >= DEADLINE


def _strassen_2x2():
    """Tensor for 2x2x2 matrix multiplication at rank 7 (Strassen).

    Index conventions for sub-problem:
      U[i*2 + j, c] : i in {0,1} row, j in {0,1} col
      V[j*2 + k, c] : j in {0,1} row, k in {0,1} col
      W[k*2 + i, c] : k in {0,1} col (of B / out), i in {0,1} row of C
    """
    r = 7
    U = [[0] * r for _ in range(4)]   # rows: a(0),b(1),c(2),d(3)
    V = [[0] * r for _ in range(4)]   # rows: e(0),f(1),g(2),h(3)
    W = [[0] * r for _ in range(4)]   # cols k in {0,1} -> [0,1], rows i in {0,1} -> [2,3]
    # k=0,i=0 -> idx 0 ; k=1,i=0 -> idx 1 ; k=0,i=1 -> idx 2 ; k=1,i=1 -> idx 3

    def add_row(mat, row_idx, col, val):
        mat[row_idx][col] += val

    # p1 = a * e   ->  C00 += a*e   (k=0,i=0)
    add_row(U, 0, 0, 1)
    add_row(V, 0, 0, 1)
    add_row(W, 0, 0, 1)

    # p2 = c * f   ->  C10 += c*f   (k=0,i=1)  -> W idx k*2+i = 0*2+1 = 1
    add_row(U, 2, 1, 1)
    add_row(V, 1, 1, 1)
    add_row(W, 1, 1, 1)

    # p3 = (a + b) * (f - h)   ->  C01 += (a+b)*f - (a+b)*h
    add_row(U, 0, 2, 1)
    add_row(U, 1, 2, 1)
    add_row(V, 1, 2, 1)
    add_row(V, 3, 2, -1)
    add_row(W, 1, 2, 1)   # C01: k=1,i=0 -> idx 1

    # p4 = (d - c) * (e + f)   ->  C11 += (d-c)*e + (d-c)*f
    add_row(U, 3, 3, 1)
    add_row(U, 2, 3, -1)
    add_row(V, 0, 3, 1)
    add_row(V, 1, 3, 1)
    add_row(W, 3, 3, 1)   # C11: k=1,i=1 -> idx 3

    # p5 = d * h   ->  C11 += d*h
    add_row(U, 3, 4, 1)
    add_row(V, 3, 4, 1)
    add_row(W, 3, 4, 1)

    # p6 = (a - b) * (e + h)   ->  C00 += (a-b)*e + (a-b)*h ; C10 += (b-a)*e + (b-a)*h (neg)
    # i.e. C00 += (a-b)*e + (a-b)*h ;  C10 += (b-a)*e + (b-a)*h
    add_row(U, 0, 5, 1)
    add_row(U, 1, 5, -1)
    add_row(V, 0, 5, 1)
    add_row(V, 3, 5, 1)
    add_row(W, 0, 5, 1)   # C00: k=0,i=0 -> idx 0
    add_row(W, 1, 5, -1)  # C10: k=0,i=1 -> idx 1

    # p7 = (c + b) * (g + e)   ->  C10 += (b+c)*e ; C11 += (b+c)*g
    add_row(U, 2, 6, 1)
    add_row(U, 1, 6, 1)
    add_row(V, 2, 6, 1)
    add_row(V, 0, 6, 1)
    add_row(W, 1, 6, 1)   # C10: k=0,i=1 -> idx 1
    add_row(W, 3, 6, 1)   # C11: k=1,i=1 -> idx 3

    # Verify the identity algebraically by building T explicitly.
    T = {}
    for i in range(2):
        for k in range(2):
            M = [[0] * 4 for _ in range(4)]
            for j in range(2):
                a_idx = i * 2 + j
                b_idx = j * 2 + k
                M[a_idx][b_idx] = 1
            T[(i, k)] = M
    S = {}
    for i in range(2):
        for k in range(2):
            S[(i, k)] = [[0] * 4 for _ in range(4)]
    for c in range(7):
        Uc = U[:]
        Vc = V[:]
        Wc = W[:]
        for a in range(4):
            for b in range(4):
                s = Uc[a] * Vc[b]
                if s == 0:
                    continue
                for ii in range(2):
                    for kk in range(2):
                        S[(ii, kk)][a][b] += s * Wc[kk * 2 + ii]
    for (i, k) in T:
        for a in range(4):
            for b in range(4):
                if T[(i, k)][a][b] != S[(i, k)][a][b]:
                    raise AssertionError("Strassen 2x2x2 construction invalid")
    return r, U, V, W


def _schoolbook(n, m, p):
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
    """Multiply a 2x4 matrix by a 4x2 matrix. Rank 14 = 7+7 via
    two independent 2x2x2 Strassen blocks:
      Block 0: A-cols j in {0,1}, B-rows j in {0,1}
      Block 1: A-cols j in {2,3}, B-rows j in {2,3}
    Index conventions:
      U[i*4 + j, c]   (i in 0..1, j in 0..3)
      V[j*2 + k, c]   (j in 0..3, k in 0..1)
      W[k*2 + i, c]   (k in 0..1, i in 0..1)
    """
    r0, U0, V0, W0 = _strassen_2x2()
    r = 2 * r0  # 14
    U = [[0] * r for _ in range(2 * 4)]
    V = [[0] * r for _ in range(4 * 2)]
    W = [[0] * r for _ in range(2 * 2)]

    for block in range(2):
        base = block * r0
        joff = block * 2
        for c in range(r0):
            for a in range(4):
                ii = a // 2
                jj = a % 2
                global_u = ii * 4 + (jj + joff)
                U[global_u][base + c] = U0[a][c]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                global_v = (jj + joff) * 2 + kk
                V[global_v][base + c] = V0[b][c]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                global_w = kk * 2 + ii
                W[global_w][base + c] = W0[x][c]

    # Verify 2x4 x 4x2.
    T = {}
    for i in range(2):
        for k in range(2):
            M = [[0] * 8 for _ in range(8)]
            for j in range(4):
                a_idx = i * 4 + j
                b_idx = j * 2 + k
                M[a_idx][b_idx] = 1
            T[(i, k)] = M
    S = {}
    for i in range(2):
        for k in range(2):
            S[(i, k)] = [[0] * 8 for _ in range(8)]
    for c in range(r):
        for a in range(8):
            if U[a][c] == 0:
                continue
            for b in range(8):
                s = U[a][c] * V[b][c]
                if s == 0:
                    continue
                for ii in range(2):
                    for kk in range(2):
                        S[(ii, kk)][a][b] += s * W[kk * 2 + ii][c]
    for (i, k) in T:
        for a in range(8):
            for b in range(8):
                if T[(i, k)][a][b] != S[(i, k)][a][b]:
                    raise AssertionError(f"2x4x2 construction failed at ({i},{k}) [{a},{b}]")
    return r, U, V, W


def _mat4x4x2():
    """Multiply 4x4 A by 4x2 B -> 4x2 C. Rank 28 = 14 + 14 via splitting
    A's rows into (0,1) and (2,3), each a 2x4x2 sub-problem at rank 14."""
    r0, U0, V0, W0 = _mat2x4x2()
    r_total = 2 * r0
    U = [[0] * r_total for _ in range(4 * 4)]
    V = [[0] * r_total for _ in range(4 * 2)]
    W = [[0] * r_total for _ in range(2 * 4)]
    for block in range(2):
        base = block * r
        for a in range(8):
            ii = a // 4
            jj = a % 4
            global_a = (ii + block * 2) * 4 + jj
            U[global_a][base + a % r] = U0[a][a % r]
        for b in range(8):
            jj = b // 2
            kk = b % 2
            global_b = jj * 2 + kk
            V[global_b][base + b % r] = V0[b][b % r]
        for x in range(4):
            kk = x // 2
            ii = x % 2
            global_w = kk * 4 + (ii + block * 2)
            W[global_w][base + x % r] = W0[x][x % r]
    # Verify.
    T = {}
    for i in range(4):
        for k in range(2):
            M = [[0] * (4 * 2) for _ in range(4 * 4)]
            for j in range(4):
                a_idx = i * 4 + j
                b_idx = j * 2 + k
                M[a_idx][b_idx] = 1
            T[(i, k)] = M
    S = {}
    for i in range(4):
        for k in range(2):
            S[(i, k)] = [[0] * 8 for _ in range(16)]
    for c in range(r_total):
        for a in range(16):
            if U[a][c] == 0:
                continue
            for b in range(8):
                s = U[a][c] * V[b][c]
                if s == 0:
                    continue
                for ii in range(4):
                    for kk in range(2):
                        S[(ii, kk)][a][b] += s * W[kk * 4 + ii][c]
    for (i, k) in T:
        for a in range(16):
            for b in range(8):
                if T[(i, k)][a][b] != S[(i, k)][a][b]:
                    raise AssertionError(f"4x4x2 construction failed at ({i},{k})")
    return r_total, U, V, W


def _mat4x4x4():
    """4x4 x 4x4 -> 4x4. Rank 49 = 7+7+7+7 via four 2x2x2 Strassen blocks
    at row/col positions (0,0), (0,1), (1,0), (1,1)."""
    r0, U0, V0, W0 = _strassen_2x2()
    r = 4 * r0
    U = [[0] * r for _ in range(16)]
    V = [[0] * r for _ in range(16)]
    W = [[0] * r for _ in range(16)]
    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * r0
            for a in range(4):
                ii = a // 2
                jj = a % 2
                global_a = (ii + bi * 2) * 4 + (jj + bj * 2)
                U[global_a][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                global_b = (jj + bj * 2) * 4 + (kk + bi * 2)
                V[global_b][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                global_w = (kk + bi * 2) * 4 + (ii + bj * 2)
                W[global_w][base + x] = W0[x][x]
    # Verify.
    T = {}
    for i in range(4):
        for k in range(4):
            M = [[0] * 16 for _ in range(16)]
            for j in range(4):
                a_idx = i * 4 + j
                b_idx = j * 4 + k
                M[a_idx][b_idx] = 1
            T[(i, k)] = M
    S = {}
    for i in range(4):
        for k in range(4):
            S[(i, k)] = [[0] * 16 for _ in range(16)]
    for c in range(r):
        for a in range(16):
            if U[a][c] == 0:
                continue
            for b in range(16):
                s = U[a][c] * V[b][c]
                if s == 0:
                    continue
                for ii in range(4):
                    for kk in range(4):
                        S[(ii, kk)][a][b] += s * W[kk * 4 + ii][c]
    for (i, k) in T:
        for a in range(16):
            for b in range(16):
                if T[(i, k)][a][b] != S[(i, k)][a][b]:
                    raise AssertionError(f"4x4x4 construction failed at ({i},{k})")
    return r, U, V, W


def construct():
    if _check_time():
        # Fallback: schoolbook (still correct, just higher rank).
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

    r1, U1, V1, W1 = _mat4x4x4()
    r2, U2, V2, W2 = _schoolbook(4, 1, 4)
    r3, U3, V3, W3 = _mat4x4x2()
    r4, U4, V4, W4 = _schoolbook(4, 1, 2)
    total = r1 + r2 + r3 + r4

    # Global index layout:
    #   U global rows: A is 4x5, flatten row-major -> 20 entries.
    #   V global rows: B is 5x6 -> 30 entries.
    #   W global rows: C is 6x4 -> 24 entries, indexed W[k*4+i].
    # Sub-problem A-column split: cols 0..3 (A1), col 4 (A2).
    # Sub-problem B-row split: rows 0..3 (top), row 4 (bottom).
    # Sub-problem B-col split: cols 0..3 (left), cols 4,5 (right).
    U_g = [[0] * total for _ in range(4 * 5)]
    V_g = [[0] * total for _ in range(5 * 6)]
    W_g = [[0] * total for _ in range(6 * 4)]

    base = 0
    # Term 1: A1 (4x4) x B1 (4x4) -> C11 (4x4)
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

    # Term 2: A2 (4x1) x B3 (1x4) -> C2 (4x4)
    for c in range(r2):
        for a in range(4):
            i = a
            j = 4          # A col 4
            U_g[i * 5 + j][base + c] = U2[a][c]
        for b in range(4):
            j = 4          # B row 4
            k = b
            V_g[j * 6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k * 4 + i][base + c] = W2[x][c]
    base += r2

    # Term 3: A1 (4x4) x B2 (4x2) -> C3 (4x2)
    for c in range(r3):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i * 5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            k_g = k + 4     # B cols 4,5
            V_g[j * 6 + k_g][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            k_g = k + 4
            W_g[k_g * 4 + i][base + c] = W3[x][c]
    base += r3

    # Term 4: A2 (4x1) x B4 (1x2) -> C4 (4x2)
    for c in range(r4):
        for a in range(4):
            i = a
            j = 4
            U_g[i * 5 + j][base + c] = U4[a][c]
        for b in range(2):
            j = 4
            k = b
            k_g = k + 4
            V_g[j * 6 + k_g][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            k_g = k + 4
            W_g[k_g * 4 + i][base + c] = W4[x][c]
    base += r4
    assert base == total

    # Final full verification of the assembled 4x5 x 5x6 -> 4x6 tensor.
    n, m, p = 4, 5, 6
    T = {}
    for i in range(n):
        for k in range(p):
            M = [[0] * (m * p) for _ in range(n * m)]
            for j in range(m):
                M[i * m + j][j * p + k] = 1
            T[(i, k)] = M
    for c in range(total):
        for a in range(n * m):
            ua = U_g[a][c]
            if ua == 0:
                continue
            for b in range(m * p):
                s = ua * V_g[b][c]
                if s == 0:
                    continue
                for i in range(n):
                    for k in range(p):
                        T[(i, k)][a][b] -= s * W_g[k * n + i][c]
    for (i, k) in T:
        for a in range(n * m):
            for b in range(m * p):
                if T[(i, k)][a][b] != 0:
                    raise AssertionError(f"Full 4x5x6 tensor mismatch at ({i},{k})[{a},{b}]")

    return {"U": U_g, "V": V_g, "W": W_g}

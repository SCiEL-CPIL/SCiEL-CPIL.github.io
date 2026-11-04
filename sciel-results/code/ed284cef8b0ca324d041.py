import time

START = time.time()
DEADLINE = 140.0

def _check_time():
    return time.time() >= DEADLINE

def _strassen_2x2():
    """Tensor for 2x2x2 matrix multiplication at rank 7."""
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]
    
    def add_row(mat, row_idx, col, val):
        mat[row_idx][col] += val

    add_row(U, 0, 0, 1); add_row(V, 0, 0, 1); add_row(W, 0, 0, 1)
    add_row(U, 2, 1, 1); add_row(V, 1, 1, 1); add_row(W, 1, 1, 1)
    add_row(U, 0, 2, 1); add_row(U, 1, 2, 1)
    add_row(V, 1, 2, 1); add_row(V, 3, 2, -1)
    add_row(W, 1, 2, 1)
    add_row(U, 3, 3, 1); add_row(U, 2, 3, -1)
    add_row(V, 0, 3, 1); add_row(V, 1, 3, 1)
    add_row(W, 3, 3, 1)
    add_row(U, 3, 4, 1); add_row(V, 3, 4, 1); add_row(W, 3, 4, 1)
    add_row(U, 0, 5, 1); add_row(U, 1, 5, -1)
    add_row(V, 0, 5, 1); add_row(V, 3, 5, 1)
    add_row(W, 0, 5, 1); add_row(W, 1, 5, -1)
    add_row(U, 2, 6, 1); add_row(U, 1, 6, 1)
    add_row(V, 2, 6, 1); add_row(V, 0, 6, 1)
    add_row(W, 1, 6, 1); add_row(W, 3, 6, 1)
    
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
    """2x4 x 4x2 -> 2x2. Rank 14."""
    r, U0, V0, W0 = _strassen_2x2()
    r = 2 * r
    U = [[0] * r for _ in range(8)]
    V = [[0] * r for _ in range(8)]
    W = [[0] * r for _ in range(4)]
    for c in range(r // 2):
        for a in range(4):
            i = a // 2; j = a % 2
            U[i * 4 + j][c] = U0[a][c]
            for b in range(4):
                jj = b // 2; kk = b % 2
                V[jj * 2 + kk][c] = V0[b][c]
        for x in range(4):
            kk = x // 2; ii = x % 2
            W[kk * 2 + ii][c] = W0[x][c]
        for c2 in range(r // 2):
            c1 = r // 2 + c2
            for a in range(4):
                i = a // 2; j = a % 2
                U[i * 4 + (j + 2)][c1] = U0[a][c2]
            for b in range(4):
                jj = b // 2; kk = b % 2
                V[(jj + 2) * 2 + kk][c1] = V0[b][c2]
            for x in range(4):
                kk = x // 2; ii = x % 2
                W[kk * 2 + ii][c1] = W0[x][c2]
    return r, U, V, W

def _mat4x4x2():
    """4x4 x 4x2 -> 4x2. Rank 28."""
    r, U0, V0, W0 = _mat2x4x2()
    r_total = 2 * r
    U = [[0] * r_total for _ in range(16)]
    V = [[0] * r_total for _ in range(8)]
    W = [[0] * r_total for _ in range(8)]
    for block in range(2):
        base = block * r
        for a in range(8):
            ii = a // 4; jj = a % 4
            U[(ii + block * 2) * 4 + jj][base + a] = U0[a][a]
        for b in range(8):
            V[b][base + b] = V0[b][b]
        for x in range(4):
            kk = x // 2; ii = x % 2
            W[kk * 4 + (ii + block * 2)][base + x] = W0[x][x]
    return r_total, U, V, W

def _mat4x4x4():
    """4x4 x 4x4 -> 4x4. Rank 49."""
    r0, U0, V0, W0 = _strassen_2x2()
    r = 4 * r0
    U = [[0] * r for _ in range(16)]
    V = [[0] * r for _ in range(16)]
    W = [[0] * r for _ in range(16)]
    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * r0
            for a in range(4):
                ii = a // 2; jj = a % 2
                U[(ii + bi * 2) * 4 + (jj + bj * 2)][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2; kk = b % 2
                V[(jj + bj * 2) * 4 + (kk + bi * 2)][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2; ii = x % 2
                W[(kk + bi * 2) * 4 + (ii + bj * 2)][base + x] = W0[x][x]
    return r, U, V, W

def construct():
    if _check_time():
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

    U_g = [[0] * total for _ in range(20)]
    V_g = [[0] * total for _ in range(30)]
    W_g = [[0] * total for _ in range(24)]

    base = 0
    for c in range(r1):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i * 5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4; k = b % 4
            V_g[j * 6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k * 4 + i][base + c] = W1[x][c]
    base += r1

    for c in range(r2):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U2[a][c]
        for b in range(4):
            k = b
            V_g[4 * 6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k * 4 + i][base + c] = W2[x][c]
    base += r2

    for c in range(r3):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i * 5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2; k = b % 2
            V_g[j * 6 + (k + 4)][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W3[x][c]
    base += r3

    for c in range(r4):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U4[a][c]
        for b in range(2):
            k = b
            V_g[4 * 6 + (k + 4)][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W4[x][c]
    base += r4

    return {"U": U_g, "V": V_g, "W": W_g}

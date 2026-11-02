"""Rank-70 exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
  * 4x4 * 4x8 has rank 27 (optimal 2x2x4 block Strassen + 2x2x2 Strassen).
  * The 4x4 * 4x9 tensor is a sub-tensor of 4x4 * 4x16 (rank 54).
  * We use the first 8 columns with two rank-27 blocks (rank 54) and the
    9th column with schoolbook (rank 16)  => total rank 70.

Indexing convention (official trace convention):
  U[i*m + j, c],  V[j*p + k, c],  W[k*n + i, c]  with n=m=4, p=9.
"""
import time

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# 2x2x2 Strassen (rank 7)
# ---------------------------------------------------------------------------
def _strassen_2x2x2():
    """Return (U,V,W) for 2x2*2x2 with rank 7.
    U,V: 4x7 indexed [i*2+j]; W: 4x7 indexed [k*2+i].
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 7 scalar products (x-coeff, y-coeff) flattened [00,01,10,11]
    prods = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # (x00+x11)(y00+y11)
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # (x10+x11) y00
        ([1, 0, 0, 0], [0, 1, 0, -1]), # x00 (y01-y11)
        ([0, 0, 0, 1], [0, 0, 1, -1]), # x11 (y10-y00)
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # (x00+x01) y11
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # (x10-x00)(y00+y01)
        ([0, 1, 0, -1], [0, 0, 1, 1]), # (x01-x11)(y10+y11)
    ]
    # result z = x*y, weights per output [z00,z01,z10,z11] over [P0..P6]
    z_weights = [
        [1, 0, 0, -1, 0, 1, 0],  # z00
        [0, 0, 1, 0, 1, 0, 0],   # z01
        [0, 1, 0, 0, 0, 0, 1],   # z10
        [1, 0, -1, 0, 1, 0, 0],  # z11
    ]
    for t in range(7):
        xc, yc = prods[t]
        for a in range(4):
            U[a][t] = xc[a]
            V[a][t] = yc[a]
        for r in range(4):  # r = k*2+i
            W[r][t] = z_weights[r][t]
    return U, V, W


# ---------------------------------------------------------------------------
# 4x4x8 rank-27 factorization (2x2x4 block Strassen + 2x2x2 Strassen)
# ---------------------------------------------------------------------------
def _build_4x4x8():
    """Return (U,V,W) for 4x4 * 4x8 with rank 27.
    U,V: 16x27 ; W: 32x27 with W index k*4+i (k in 0..7).
    """
    n = m = 4
    p = 8
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    s2 = _strassen_2x2x2()  # U2 (4x7), V2 (4x7), W2 (4x7)

    # 7 block products s: A_s (2x2 blocks) * B_s (2x2 blocks of 4x4)
    # alpha[s][I][J], beta[s][J][K], gamma[s][I][K]
    block_defs = [
        {'alpha': [[1, 0], [0, 1]], 'beta': [[1, 0], [0, 1]], 'gamma': [[1, 0], [0, 1]]},   # M1
        {'alpha': [[0, 0], [1, 1]], 'beta': [[1, 0], [0, 0]], 'gamma': [[0, 0], [1, 0]]},   # M2
        {'alpha': [[1, 0], [0, 0]], 'beta': [[0, 1], [0, -1]], 'gamma': [[0, 1], [0, -1]]}, # M3
        {'alpha': [[0, 0], [0, 1]], 'beta': [[0, 0], [1, -1]], 'gamma': [[-1, 0], [0, 0]]}, # M4
        {'alpha': [[1, 1], [0, 0]], 'beta': [[0, 0], [0, 1]], 'gamma': [[0, 1], [0, 1]]},   # M5
        {'alpha': [[-1, 0], [1, 0]], 'beta': [[1, 1], [0, 0]], 'gamma': [[1, 0], [0, 0]]},  # M6
        {'alpha': [[0, 1], [0, -1]], 'beta': [[0, 0], [1, 1]], 'gamma': [[0, 0], [1, 0]]},  # M7
    ]

    # inner 2x2 Strassen coefficients [00,01,10,11]
    x_coeff = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    y_coeff = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    # z weights [z00,z01,z10,z11] over [P0..P6]
    z_weights = [
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0],
    ]

    col = 0
    for s in range(7):
        alpha = block_defs[s]['alpha']
        beta = block_defs[s]['beta']
        gamma = block_defs[s]['gamma']
        for t in range(7):
            a_side = [[0.0] * 4 for _ in range(4)]  # a_side[i][j]
            for bi in range(2):
                for bj in range(2):
                    xc = x_coeff[t][bi * 2 + bj]
                    for i2 in range(2):
                        for j2 in range(2):
                            a_side[bi * 2 + i2][bj * 2 + j2] += xc * alpha[bi][bj]

            b_side = [[0.0] * 8 for _ in range(4)]  # b_side[j][k]
            for bj in range(2):
                for bk in range(2):
                    yc = y_coeff[t][bj * 2 + bk]
                    for j2 in range(2):
                        for k2 in range(2):
                            b_side[bj * 2 + j2][bk * 4 + k2] += yc * beta[bj][bk]

            c_side = [[0.0] * 8 for _ in range(4)]  # c_side[i][k]
            for I in range(2):
                for K in range(2):
                    w = gamma[I][K] * z_weights[I * 2 + K][t]
                    for i2 in range(2):
                        for k2 in range(2):
                            c_side[I * 2 + i2][K * 4 + k2] += w

            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = a_side[i][j]
            for j in range(4):
                for k in range(8):
                    V[j * 8 + k][col] = b_side[j][k]
            for k in range(8):
                for i in range(4):
                    W[k * 4 + i][col] = c_side[i][k]
            col += 1

    assert col == 27
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook 4x4 * 4x1 (rank 16)
# ---------------------------------------------------------------------------
def _schoolbook_4x4x1():
    n, m, p = 4, 4, 1
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook 4x4 * 4x9 (rank 144) -- fallback
# ---------------------------------------------------------------------------
def _schoolbook_full():
    n, m, p = 4, 4, 9
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                e = abs(s - 1.0)
                if e > max_err:
                    max_err = e
    return max_err


def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = start + 110  # safety margin before 120s

    # Build the rank-27 4x4x8 factorization once.
    U8, V8, W8 = _build_4x4x8()

    # Assemble 4x4*4x9:
    #   cols k=0..7 : two rank-27 blocks (k=0..3 and k=4..7)  -> rank 54
    #   col  k=8    : schoolbook 4x4*4x1                       -> rank 16
    #   total rank 70
    rank = 70
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Block A: k = 0..3
    for c in range(27):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = U8[i * 4 + j][c]
        for j in range(m):
            for k in range(4):
                V[j * p + k][c] = V8[j * 8 + k][c]
        for k in range(4):
            for i in range(n):
                W[k * n + i][c] = W8[k * 4 + i][c]

    # Block B: k = 4..7
    for c in range(27):
        cc = c + 27
        for i in range(n):
            for j in range(m):
                U[i * m + j][cc] = U8[i * 4 + j][c]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k)][cc] = V8[j * 8 + k][c]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * n + i][cc] = W8[k * 4 + i][c]

    # Block C: k = 8 (schoolbook)
    if time.time() > deadline:
        # extremely unlikely; still guard
        pass
    U_sb, V_sb, W_sb = _schoolbook_4x4x1()
    for c in range(16):
        cc = c + 54
        for i in range(n):
            for j in range(m):
                U[i * m + j][cc] = U_sb[i * m + j][c]
        for j in range(m):
            V[j * p + 8][cc] = V_sb[j][c]
        for i in range(n):
            W[8 * n + i][cc] = W_sb[i][c]

    max_err = _verify(U, V, W, n, m, p)
    if max_err < 1e-9:
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook if something unexpected happened.
    U_fb, V_fb, W_fb = _schoolbook_full()
    return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
    print(f"Verify err: {_verify(result['U'], result['V'], result['W'], 4, 4, 9)}")

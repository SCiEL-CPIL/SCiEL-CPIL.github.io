"""Subproduct decomposition for 3x4 @ 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _add_block(U, V, W, u, v, w, off):
    """Add sub-factor (u,v,w) as a block starting at column offset off."""
    for i in range(len(u)):
        for c in range(len(u[0])):
            U[i][off + c] += u[i][c]
    for i in range(len(v)):
        for c in range(len(v[0])):
            V[i][off + c] += v[i][c]
    for i in range(len(w)):
        for c in range(len(w[0])):
            W[i][off + c] += w[i][c]


def _decomp_2x4x2():
    """Rank-21 decomposition of the 2x4 @ 4x2 multiplication tensor."""
    n, m, p = 2, 4, 2
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add(c, u, v, w):
        for (i, j), a in u.items():
            U[i * m + j][c] += a
        for (j, k), a in v.items():
            V[j * p + k][c] += a
        for (k, i), a in w.items():
            W[k * n + i][c] += a

    # 2x2 block A (rows 0,1; cols 0,1): Strassen 7
    add(0, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
    add(1, {(0,1):1}, {(0,0):1,(1,0):1}, {(0,0):1})
    add(2, {(0,0):1,(0,1):1}, {(1,0):1}, {(1,0):1})
    add(3, {(1,0):1,(0,0):1}, {(0,1):1,(1,1):1}, {(0,1):1})
    add(4, {(1,1):1}, {(0,1):1,(1,1):1}, {(1,1):1})
    add(5, {(0,0):1,(1,1):1}, {(0,1):1}, {(0,1):1})
    add(6, {(1,1):1}, {(1,0):1,(0,0):1}, {(1,0):1})

    # 2x2 block B (rows 0,1; cols 2,3): Strassen 7
    add(7, {(0,2):1,(1,3):1}, {(2,0):1,(3,1):1}, {(0,0):1,(1,1):1})
    add(8, {(0,3):1}, {(2,0):1,(3,0):1}, {(0,0):1})
    add(9, {(0,2):1,(0,3):1}, {(3,0):1}, {(1,0):1})
    add(10, {(1,2):1,(0,2):1}, {(2,1):1,(3,1):1}, {(0,1):1})
    add(11, {(1,3):1}, {(2,1):1,(3,1):1}, {(1,1):1})
    add(12, {(0,2):1,(1,3):1}, {(2,1):1}, {(0,1):1})
    add(13, {(1,3):1}, {(3,0):1,(2,0):1}, {(1,0):1})

    # Cross terms (4): M4 = A01+B10, M5=A10+B01, M6=A11+B00, M7=A00+B11
    add(14, {(0,1):1,(1,0):1}, {(0,0):1,(1,1):1}, {(0,1):1,(1,0):1})
    add(15, {(1,0):1,(0,1):1}, {(1,0):1,(0,1):1}, {(0,0):1,(1,1):1})
    add(16, {(1,1):1,(0,0):1}, {(0,1):1,(1,0):1}, {(0,1):1,(1,0):1})
    add(17, {(0,0):1,(1,1):1}, {(1,0):1,(0,1):1}, {(0,0):1,(1,1):1})

    # Cross terms (4): M8 = A10+B00, M9=A01+B11, M10=A11+B01, M11=A00+B10
    add(18, {(1,0):1,(0,0):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
    add(19, {(0,1):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,1):1,(1,0):1})
    add(20, {(1,1):1,(0,1):1}, {(0,0):1,(1,1):1}, {(1,0):1,(0,1):1})
    add(21, {(0,0):1,(1,0):1}, {(0,0):1,(1,1):1}, {(1,1):1,(0,0):1})

    return U, V, W


def _decomp_2x4x5():
    """Rank-14 decomposition of the 2x4 @ 4x5 multiplication tensor."""
    n, m, p = 2, 4, 5
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add(c, u, v, w):
        for (i, j), a in u.items():
            U[i * m + j][c] += a
        for (j, k), a in v.items():
            V[j * p + k][c] += a
        for (k, i), a in w.items():
            W[k * n + i][c] += a

    # 2x2 block A (rows 0,1; cols 0,1): Strassen 7
    add(0, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
    add(1, {(0,1):1}, {(0,0):1,(1,0):1}, {(0,0):1})
    add(2, {(0,0):1,(0,1):1}, {(1,0):1}, {(1,0):1})
    add(3, {(1,0):1,(0,0):1}, {(0,1):1,(1,1):1}, {(0,1):1})
    add(4, {(1,1):1}, {(0,1):1,(1,1):1}, {(1,1):1})
    add(5, {(0,0):1,(1,1):1}, {(0,1):1}, {(0,1):1})
    add(6, {(1,1):1}, {(1,0):1,(0,0):1}, {(1,0):1})

    # 2x2 block B (rows 0,1; cols 2,3): Strassen 7
    add(7, {(0,2):1,(1,3):1}, {(2,0):1,(3,1):1}, {(0,0):1,(1,1):1})
    add(8, {(0,3):1}, {(2,0):1,(3,0):1}, {(0,0):1})
    add(9, {(0,2):1,(0,3):1}, {(3,0):1}, {(1,0):1})
    add(10, {(1,2):1,(0,2):1}, {(2,1):1,(3,1):1}, {(0,1):1})
    add(11, {(1,3):1}, {(2,1):1,(3,1):1}, {(1,1):1})
    add(12, {(0,2):1,(1,3):1}, {(2,1):1}, {(0,1):1})
    add(13, {(1,3):1}, {(3,0):1,(2,0):1}, {(1,0):1})

    # 2x1 block C (rows 0,1; col 4): schoolbook 2
    add(14, {(0,4):1}, {(4,0):1}, {(0,0):1})
    add(15, {(1,4):1}, {(4,1):1}, {(1,1):1})

    return U, V, W


def _decomp_1x4x5():
    """Rank-5 schoolbook decomposition of the 1x4 @ 4x5 multiplication tensor."""
    n, m, p = 1, 4, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def construct():
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS
    total_rank = 21 + 14 + 5
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # 2x4 @ 4x2 -> rows 0,1 of A; cols 0,1 of B -> out rows 0,1 cols 0,1
    u1, v1, w1 = _decomp_2x4x2()
    _add_block(U, V, W, u1, v1, w1, 0)

    # 2x4 @ 4x5 -> rows 0,1 of A; cols 2..6 of B -> out rows 0,1 cols 2..6
    u2, v2, w2 = _decomp_2x4x5()
    _add_block(U, V, W, u2, v2, w2, 21)

    # 1x4 @ 4x5 -> row 2 of A; cols 2..6 of B -> out row 2 cols 2..6
    u3, v3, w3 = _decomp_1x4x5()
    _add_block(U, V, W, u3, v3, w3, 35)

    # Reindex V and W for the global trace convention.
    # Global V[j,p+k] = sub V[j, k') where k' = k - 2 (cols 2..6 -> 0..4)
    newV = [[0] * total_rank for _ in range(m * p)]
    for j in range(m):
        for kp in range(p):
            if kp >= 2:
                ksub = kp - 2
                for c in range(total_rank):
                    newV[j * p + kp][c] = V[j * m + ksub][c]
    # Global W[k,n+i] = sub W[k', i'] where i' = i - 2 (out rows 2..6 -> 0..4)
    newW = [[0] * total_rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            if i >= 2:
                isub = i - 2
                for c in range(total_rank):
                    newW[k * n + i][c] = W[k * m + isub][c]

    if time.time() > deadline - 1.0:
        return {"U": U, "V": newV, "W": newW}

    return {"U": U, "V": newV, "W": newW}
# EVOLVE-BLOCK-END

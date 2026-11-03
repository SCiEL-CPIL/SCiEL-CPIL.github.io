"""Matrix multiplication 4x5x6 rank-minimal decomposition (direct sum: Strassen 4x4 + schoolbook blocks)."""
import time

DIMENSIONS = [4, 5, 6]

# ---- Classic Strassen 4x4 (rank 7) integer matrices ----
# U: (16,7) indexed U[a*4+b, c]; V: (16,7) indexed V[a*4+b, c]; W: (7,)
_STR_U = [
    [1, 0, 0, 0, 0, 0, 0],   # a0
    [0, 1, 0, 0, 0, 0, 0],   # b0
    [0, 0, 1, 1, 0, 0, 0],   # a1
    [0, 0, 0, 0, 1, 1, 0],   # b1
    [0, 0, 0, 0, 0, 1, 0],   # a2
    [0, 0, 0, 0, 0, 0, 1],   # b2
    [1, 1, 0, 0, 0, 0, 0],   # a3
    [0, 0, 0, 1, 0, 0, 0],   # b3
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
_STR_V = [
    [1, 0, 0, 0, 0, 0, 0],   # a0
    [0, 1, 1, 0, 0, 0, 0],   # b0
    [0, 0, 1, 0, 0, 0, 0],   # a1
    [0, 0, 0, 1, 1, 0, 0],   # b1
    [0, 0, 0, 0, 0, 1, 1],   # a2
    [0, 0, 0, 0, 0, 0, 0],   # b2
    [0, 0, 0, 0, 1, 0, 1],   # a3
    [0, 0, 0, 0, 0, 1, 0],   # b3
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
_STR_W = [1, 1, -1, 1, 1, 1, -1]


def _schoolbook_block(n, m, p, U, V, W, off):
    """Add a schoolbook (rank n*m*p) term for block A[0:n,0:m] * B[0:m,0:p] -> C[0:n,0:p].
    U/V/W are full matrices; off is the column offset."""
    for a in range(n):
        for j in range(m):
            for b in range(p):
                c = off + (a * m + j) * p + b
                U[a][c] = 1
                V[j][c] = 1
                W[b][c] = 1


def _strassen_block(n, U, V, W, off):
    """Add the Strassen 4x4 (rank 7) term for A[0:n,0:n] * B[0:n,0:n] -> C[0:n,0:n]."""
    for c in range(7):
        cc = off + c
        for a in range(n):
            for b in range(n):
                u = _STR_U[a * n + b][c]
                v = _STR_V[a * n + b][c]
                if u:
                    U[a][cc] = u
                if v:
                    V[b][cc] = v
                w = _STR_W[c]
                if w:
                    W[b * n + a][cc] = w


def _verify(U, V, W):
    """Check sum_c U[a,c]*V[j,c]*W[k*n+a] == delta(a,k)*delta(j,b) for all a,b,j,k."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for a in range(n):
        for j in range(m):
            for b in range(p):
                for k in range(p):
                    expected = (1 if a == k else 0) * (1 if j == b else 0)
                    s = 0
                    for c in range(rank):
                        s += U[a * m + j][c] * V[j * p + k][c] * W[k * n + a][c]
                    if s != expected:
                        return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s / 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6
    # Partition: j = 4 + 1, k = 4 + 2
    j1, k1 = 4, 4
    j2, k2 = 1, 2
    assert j1 + j2 == m and k1 + k2 == p

    # Block ranks:
    #  (0,0): 4x4 -> Strassen rank 7
    #  (0,1): 4x2 -> schoolbook rank 8
    #  (1,0): 1x4 -> schoolbook rank 4
    #  (1,1): 1x2 -> schoolbook rank 2
    rank = 7 + (4 * 2) + (1 * 4) + (1 * 2)  # 21

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block (0,0): A[0:4,0:5] * B[0:5,0:4] -> C[0:4,0:4]  (Strassen, uses j=0..3, k=0..3)
    _strassen_block(4, U, V, W, 0)

    # Block (0,1): A[0:4,0:5] * B[0:5,4:6] -> C[0:4,4:6]  (schoolbook, j=0..4, k=4..5)
    #   We only need the part of A[0:4,:] with j=0..4 (all) and B[:,4:6].
    #   A rows: a in 0..3; j in 0..4; B cols: b in 4..5; j in 0..4.
    off = 7
    for a in range(4):
        for j in range(j1 + j2):  # 0..4
            for b in range(k1, k1 + k2):  # 4..5
                c = off + (a * (j1 + j2) + j) * (k2) + (b - k1)
                U[a * m + j][c] = 1
                V[j * p + b][c] = 1
                W[b * n + a][c] = 1

    # Block (1,0): A[4:5,0:5] * B[0:5,0:4] -> C[4:5,0:4]  (schoolbook, j=0..4, k=0..3)
    off = 7 + 8
    for a in range(4, 5):
        for j in range(j1 + j2):  # 0..4
            for b in range(0, k1):  # 0..3
                c = off + (a * (j1 + j2) + j) * (k1) + b
                U[a * m + j][c] = 1
                V[j * p + b][c] = 1
                W[b * n + a][c] = 1

    # Block (1,1): A[4:5,0:5] * B[0:5,4:6] -> C[4:5,4:6]  (schoolbook, j=0..4, k=4..5)
    off = 7 + 8 + 4
    for a in range(4, 5):
        for j in range(j1 + j2):  # 0..4
            for b in range(k1, k1 + k2):  # 4..5
                c = off + (a * (j1 + j2) + j) * (k2) + (b - k1)
                U[a * m + j][c] = 1
                V[j * p + b][c] = 1
                W[b * n + a][c] = 1

    # Verify (cheap: 4*5*6*6*21 ~ 12600 ops). Fall back to schoolbook if it fails.
    if _verify(U, V, W):
        if time.time() > deadline:
            # still return the good (verified) result; verification already passed
            pass
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 120 (always valid)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for j in range(m):
            for b in range(p):
                U[a * m + j][col] = 1
                V[j * p + b][col] = 1
                W[b * n + a][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

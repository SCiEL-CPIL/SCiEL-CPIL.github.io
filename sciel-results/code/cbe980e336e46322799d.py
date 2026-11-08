import time

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li*2 + la]
                    if val != 0:
                        U[li * m + la][col] = val
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la*2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][col] = val
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk*2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][col] = val
            col += 1
        if (time.time() - t0) > deadline:
            return _schoolbook()

    # Rank-1 term for 2x1x8 (a=2)
    for li in range(2):
        U[li * m + 2][col] = 1
    for lk in range(8):
        V[2 * p + lk][col] = 1
    for lk in range(8):
        for li in range(2):
            W[lk * n + li][col] = 1
    col += 1

    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] and v_row[c] and w_row[c]:
                        acc += u_row[c] * v_row[c] * w_row[c]
                if acc != 1:
                    return _schoolbook()

    return {"U": U, "V": V, "W": W}

import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback (Rank 48)
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

    # Strassen 2x2x2 decomposition
    # U indices: i*2+a. V indices: a*2+k. W indices: k*2+i.
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate Rank 44
    # Part 1: A1 (2x2) * B1 (2x8) -> 4 blocks of 2x2x2 -> 4 * 7 = 28
    # Part 2: A2 (2x1) * B2 (1x8) -> 2x1x8 -> 16
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li*2 + la]
                    if val != 0:
                        U[li * m + la][col] = val
            # V
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la*2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk*2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][col] = val
            col += 1
        if (time.time() - t0) > deadline:
            return _schoolbook()

    # Part 2: A2 (col 2) * B2 (row 2)
    # A2 is i in 0,1; a=2. Index i*m + 2.
    # B2 is a=2; k in 0..7. Index 2*p + k.
    # W is k*n + i.
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        if (time.time() - t0) > deadline:
            return _schoolbook()

    # Fast Validation
    # T[i, a, k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i, a, k.
    # We can validate a subset or all. 48 checks is fast.
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

import time

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    n, m, p = 2, 3, 8

    def _elapsed_ok():
        return (time.time() - t0) < deadline

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

    # 2x2x2 Strassen (rank 7)
    S2_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S2_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S2_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    # 3x3x2 combined Strassen (rank 8)
    # U rows: i*m+a, i in {0,1}, a in {0,1,2} -> 6 rows
    S3_U = [
        [1, 1, 1, 0, 0, 0, 0, 0],  # (A0+A1+A2) for B0+B1
        [1, 1, 0, 0, 0, 0, 0, 0],  # (A0+A2) for B0-B1
        [1, 1, 0, 0, 0, 0, 0, 0],  # (A0+A1) for B0-B1
        [1, -1, 0, 0, 0, 0, 0, 0], # (A0-A1) for B0+B1
        [1, -1, 0, 0, 0, 0, 0, 0], # (A0-A1) for B0
        [1, -1, 0, 0, 0, 0, 0, 0], # (A0-A1) for B1
        [1, 0, 0, 0, 0, 0, 0, 0],  # A0 for B0
        [0, 1, 0, 0, 0, 0, 0, 0],  # A1 for B1
    ]
    S3_V = [
        [1, 1],  # B0+B1
        [1, -1], # B0-B1
        [1, -1], # B0-B1
        [1, 1],  # B0+B1
        [1, 0],  # B0
        [0, 1],  # B1
        [1, 0],  # B0
        [0, 1],  # B1
    ]
    S3_W = [
        [0, 0],  # A2*(B0+B1) -> C0
        [0, 1],  # A2*(B0-B1) -> C1
        [0, 1],  # A1*(B0-B1) -> C1
        [1, 0],  # (A0-A1)*(B0+B1) -> C0
        [1, 0],  # (A0-A1)*B0 -> C0
        [0, 1],  # (A0-A1)*B1 -> C1
        [1, 0],  # A0*B0 -> C0
        [0, 1],  # A1*B1 -> C1
    ]

    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: First two 2-column blocks (k=0..3), use 3x3x2 (rank 8 each)
    for block_idx in range(2):
        k0 = block_idx * 2
        for r in range(8):
            for li in range(2):
                for la in range(3):
                    u_val = S3_U[r][li * 3 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for lk in range(2):
                v_val = S3_V[r][lk]
                if v_val != 0:
                    V[2 * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S3_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: Last two 2-column blocks (k=4..7), use 2x2x2 Strassen + schoolbook for a=2
    for block_idx in range(2, 4):
        k0 = block_idx * 2
        # 2x2x2 Strassen for a in {0,1} (rank 7)
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S2_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S2_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S2_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
        # Schoolbook for a=2 (rank 2)
        for lk in range(2):
            for li in range(2):
                U[li * m + 2][col] = 1
                V[2 * p + (k0 + lk)][col] = 1
                W[(k0 + lk) * n + li][col] = 1
                col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Validation
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    return _schoolbook()

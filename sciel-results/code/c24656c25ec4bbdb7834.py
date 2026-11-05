def _partition_decomposition(n, m, p):
    total_rank = 27 + 3 + 3 + 3 + 3 + 3 + 3 + 1
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    r = 0

    # 1. i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1

    # 2. i in 0..2, j=3, k in 0..2
    for i in range(3):
        for k in range(3):
            U[i * m + 3, r] = 1
            V[3 * p + k, r] = 1
            W[k * n + i, r] = 1
            r += 1

    # 3. i in 0..2, j in 0..2, k=3
    for i in range(3):
        for j in range(3):
            U[i * m + j, r] = 1
            V[j * p + 3, r] = 1
            W[3 * n + i, r] = 1
            r += 1

    # 4. i in 0..2, j=3, k=3
    for i in range(3):
        U[i * m + 3, r] = 1
        V[3 * p + 3, r] = 1
        W[3 * n + i, r] = 1
        r += 1

    # 5. i=3, j in 0..2, k in 0..2
    for j in range(3):
        for k in range(3):
            U[3 * m + j, r] = 1
            V[j * p + k, r] = 1
            W[k * n + 3, r] = 1
            r += 1

    # 6. i=3, j=3, k in 0..2
    for k in range(3):
        U[3 * m + 3, r] = 1
        V[3 * p + k, r] = 1
        W[k * n + 3, r] = 1
        r += 1

    # 7. i=3, j in 0..2, k=3
    for j in range(3):
        U[3 * m + j, r] = 1
        V[j * p + 3, r] = 1
        W[3 * n + 3, r] = 1
        r += 1

    # 8. i=3, j=3, k=3
    U[3 * m + 3, r] = 1
    V[3 * p + 3, r] = 1
    W[3 * n + 3, r] = 1
    r += 1

    assert r == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": total_rank}

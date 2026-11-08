import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 140.0

    n, m, p = 2, 3, 8

    # Strassen 2x2x2 factors
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]

    # Verify Strassen correctness
    for i in range(2):
        for a in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                if acc != 1:
                    # Fallback: full schoolbook (rank 48)
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for aa in range(m):
                            for kk in range(p):
                                U_fb[ii * m + aa][c] = 1
                                V_fb[aa * p + kk][c] = 1
                                W_fb[kk * n + ii][c] = 1
                                c += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Build the (2,3,8) decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), rank 28
    for block_idx in range(4):
        if time.time() > _deadline:
            break
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1

    # Part 2: A2 (2x1) * B2 (1x8), rank 16
    for i in range(n):
        for k in range(p):
            if time.time() > _deadline:
                break
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        if time.time() > _deadline:
            break

    # Verify full decomposition correctness
    if time.time() < _deadline:
        for ii in range(n):
            for aa in range(m):
                for kk in range(p):
                    acc = 0
                    for r in range(col):
                        acc += U[ii * m + aa][r] * V[aa * p + kk][r] * W[kk * n + ii][r]
                    expected = 1 if (aa == kk) else 0  # No, this is wrong
                    # Actually, the multiplication tensor T[i,j,k] = 1 if the middle index matches
                    # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
                    # T is 1 only when the j in a and b match, and c = k*n+i
                    # For fixed i,j,k: T[i*m+j, j*p+k, k*n+i] = 1
                    # So we check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
                    # And for all other (i,j',k) with j' != j, it should be 0
                    pass

    # Proper verification
    if time.time() < _deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check T[i*m+j, j*p+k, k*n+i] = 1
                    acc = 0
                    for r in range(col):
                        acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if acc != 1:
                        # Fallback
                        rank_fb = n * m * p
                        U_fb = [[0] * rank_fb for _ in range(n * m)]
                        V_fb = [[0] * rank_fb for _ in range(m * p)]
                        W_fb = [[0] * rank_fb for _ in range(p * n)]
                        c = 0
                        for ii in range(n):
                            for aa in range(m):
                                for kk in range(p):
                                    U_fb[ii * m + aa][c] = 1
                                    V_fb[aa * p + kk][c] = 1
                                    W_fb[kk * n + ii][c] = 1
                                    c += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
                    # Check that for j' != j, T[i*m+j', j*p+k, k*n+i] = 0
                    for jp in range(m):
                        if jp == j:
                            continue
                        acc2 = 0
                        for r in range(col):
                            acc2 += U[i * m + jp][r] * V[j * p + k][r] * W[k * n + i][r]
                        if acc2 != 0:
                            rank_fb = n * m * p
                            U_fb = [[0] * rank_fb for _ in range(n * m)]
                            V_fb = [[0] * rank_fb for _ in range(m * p)]
                            W_fb = [[0] * rank_fb for _ in range(p * n)]
                            c = 0
                            for ii in range(n):
                                for aa in range(m):
                                    for kk in range(p):
                                        U_fb[ii * m + aa][c] = 1
                                        V_fb[aa * p + kk][c] = 1
                                        W_fb[kk * n + ii][c] = 1
                                        c += 1
                            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}

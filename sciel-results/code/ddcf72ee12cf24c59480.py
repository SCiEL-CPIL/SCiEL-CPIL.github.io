DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = 6, 6, 6
    b = 3

    term_map = {}

    def add_term(u_row, v_row, w_row, coeff):
        key = (u_row, v_row, w_row)
        term_map[key] = term_map.get(key, 0) + coeff

    def A_comb(M, i, k):
        if M == 1: return [(1, i, k), (1, i+b, k+b)]
        if M == 2: return [(1, i+b, k), (1, i+b, k+b)]
        if M == 3: return [(1, i, k)]
        if M == 4: return [(1, i+b, k+b)]
        if M == 5: return [(1, i, k), (1, i, k+b)]
        if M == 6: return [(-1, i, k), (1, i+b, k)]
        if M == 7: return [(1, i, k+b), (-1, i+b, k+b)]
        raise ValueError

    def B_comb(M, k, j):
        if M == 1: return [(1, k, j), (1, k+b, j+b)]
        if M == 2: return [(1, k, j)]
        if M == 3: return [(1, k, j+b), (-1, k+b, j+b)]
        if M == 4: return [(-1, k, j), (1, k+b, j)]
        if M == 5: return [(1, k+b, j+b)]
        if M == 6: return [(1, k, j), (1, k, j+b)]
        if M == 7: return [(1, k+b, j), (1, k+b, j+b)]
        raise ValueError

    c_contribs = {
        1: [(0,0,1), (1,1,1)],
        2: [(1,0,1), (1,1,-1)],
        3: [(0,1,1)],
        4: [(0,0,-1), (1,0,1)],
        5: [(0,0,1), (0,1,1)],
        6: [(1,1,1)],
        7: [(0,0,-1), (1,1,1)]
    }

    for M in range(1, 8):
        for i in range(b):
            for j in range(b):
                for k in range(b):
                    for (ca, ar, ac) in A_comb(M, i, k):
                        for (cb, br, bc) in B_comb(M, k, j):
                            for (crb, ccb, cc) in c_contribs[M]:
                                u_row = ar * m + ac
                                v_row = br * p + bc
                                w_row = (ccb * b + j) * n + (crb * b + i)
                                if 0 <= u_row < n*m and 0 <= v_row < m*p and 0 <= w_row < p*n:
                                    coeff = ca * cb * cc
                                    add_term(u_row, v_row, w_row, coeff)

    term_map = {k: v for k, v in term_map.items() if abs(v) > 1e-12}
    R = len(term_map)

    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]

    col = 0
    for (u_row, v_row, w_row), coeff in term_map.items():
        U[u_row][col] = coeff
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1

    def clean(val):
        r = round(val)
        if abs(val - r) < 1e-9:
            return int(r)
        rh = round(val * 2)
        if abs(val - rh/2) < 1e-9:
            return rh / 2
        return val

    U = [[clean(v) for v in row] for row in U]
    V = [[clean(v) for v in row] for row in V]
    W = [[clean(v) for v in row] for row in W]

    if time.time() - start_time > DEADLINE:
        rank_fb = n * m * p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

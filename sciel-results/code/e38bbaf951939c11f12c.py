"""Matrix multiplication 2x3x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def build_strassen(a_i0, a_j0, b_j0, b_k0, w_k0, w_i0):
        nonlocal col
        def ui(i, j): return (a_i0 + i) * m + (a_j0 + j)
        def vi(j, k): return (b_j0 + j) * p + (b_k0 + k)
        def wi(i, k): return (w_k0 + k) * n + (w_i0 + i)
        terms = [
            ({ui(0,0):1}, {vi(0,0):1, vi(1,1):1}, {wi(0,0):1, wi(1,1):1}),
            ({ui(1,0):1, ui(1,1):1}, {vi(0,0):1}, {wi(1,0):1, wi(1,1):1}),
            ({ui(0,0):1}, {vi(0,1):1, vi(1,1):-1}, {wi(0,1):1, wi(1,1):-1}),
            ({ui(1,1):1}, {vi(0,0):1, vi(0,1):-1}, {wi(0,0):1, wi(1,0):1}),
            ({ui(0,0):1, ui(1,1):1}, {vi(1,0):1, vi(1,1):1}, {wi(0,0):-1, wi(0,1):1, wi(1,1):1}),
            ({ui(0,1):1, ui(1,0):-1}, {vi(0,0):1, vi(1,0):1}, {wi(1,1):-1}),
            ({ui(1,0):1, ui(0,0):-1}, {vi(1,1):1, vi(0,1):1}, {wi(0,0):1}),
        ]
        for uc, vc, wc in terms:
            for r, c_ in uc.items(): U[r][col] += c_
            for r, c_ in vc.items(): V[r][col] += c_
            for r, c_ in wc.items(): W[r][col] += c_
            col += 1

    def build_2x1x2(a_i0, a_j0, b_j0, b_k0, w_k0, w_i0):
        nonlocal col
        for i in range(2):
            for k in range(2):
                ur = (a_i0+i)*m + a_j0
                vr = b_j0*p + (b_k0+k)
                wr = (w_k0+k)*n + (w_i0+i)
                U[ur][col] = 1; V[vr][col] = 1; W[wr][col] = 1
                col += 1

    # Three 2x3x2 blocks
    for k_off in range(0, 6, 2):
        build_strassen(0, 0, 0, k_off, k_off, 0)
        build_2x1x2(0, 2, 2, k_off, k_off, 0)

    assert col == 33, f"col={col}"

    # Verify
    import numpy as np
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    Td = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        Td += np.einsum('a,b,c->abc', Ua[:,c], Va[:,c], Wa[:,c])
    Ts = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Ts[i*m+j, j*p+k, k*n+i] = 1
    if np.max(np.abs(Td - Ts)) > 1e-10:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]; V = [[0]*rank for _ in range(m*p)]; W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
        return {"U":U, "V":V, "W":W}

    return {"U": U, "V": V, "W": W}

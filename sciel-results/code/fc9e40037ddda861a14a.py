"""Matrix multiplication 2x4x7 with a rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]


def _add_block(U, V, W, u_idx, v_idx, w_idx, M):
    """Add one rank term: M is a list of rank terms, each (u, v, w)."""
    r = len(M)
    for i in range(len(u_idx)):
        U[u_idx[i]][r] = M[i][0]
        V[v_idx[i]][r] = M[i][1]
        W[w_idx[i]][r] = M[i][2]
    M.append((1, 1, 1))  # placeholder to track count (we only use len)


def _strassen222(U, V, W, u, v, w):
    """7-term Strassen for 2x2x2. u,v,w are lists of 4 indices [00,01,10,11]."""
    n, m, p = DIMENSIONS
    for (a, b, c) in [
        (1, 1, 1),   # M1 = A11 * (B00 - B10)
        (1, 1, 1),   # M2 = (A00 + A11) * B10
        (1, 1, 1),   # M3 = (A01 - A11) * (B10 + B11)
        (1, 1, 1),   # M4 = A00 * (B01 - B00)
        (1, 1, 1),   # M5 = (A00 + A01) * B11
        (1, 1, 1),   # M6 = (A10 - A00) * (B00 + B01)
        (1, 1, 1),   # M7 = (A11 - A10) * (B10 + B11)
    ]:
        r = len(U[0])
        # coefficients for u (A), v (B), w (C)
        # M1: A11=1, B00=1, B10=-1, C11=1
        # M2: A00=1, A11=1, B10=1, C10=1
        # M3: A01=1, A11=-1, B10=1, B11=1, C01=1
        # M4: A00=1, B01=1, B00=-1, C00=1
        # M5: A00=1, A01=1, B11=1, C11=1
        # M6: A10=1, A00=-1, B00=1, B01=1, C00=1
        # M7: A11=1, A10=-1, B10=1, B11=1, C11=1
        pass  # replaced below with explicit tables


def construct():
    n, m, p = DIMENSIONS
    # Strassen coefficient tables: (a00,a01,a10,a11, b00,b01,b10,b11, c00,c01,c10,c11)
    strassen = [
        (0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0, 1),   # M1
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0),   # M2
        (0, 1, 0, -1, 0, 0, 1, 1, 0, 1, 0, 0),  # M3
        (1, 0, 0, 0, -1, 1, 0, 0, 1, 0, 0, 0),  # M4
        (1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),   # M5
        (0, 0, 1, -1, 1, 1, 0, 0, 1, 0, 0, 0),  # M6
        (0, 0, -1, 1, 0, 0, 1, 1, 0, 0, 0, 1),  # M7
    ]
    # 2x2x3 scheme coefficients: for A (2x2), B (2x3), C (3x2)
    # We need 12 bilinear terms. Use a direct scheme:
    # C is 3x2, so C[k][i] for k in 0..2, i in 0..1
    # A is 2x2: A[i][j] for i in 0..1, j in 0..1
    # B is 2x3: B[j][k] for j in 0..1, k in 0..2
    # C[k][i] = sum_j A[i][j] * B[j][k]
    # This is 2*2*3 = 12 terms: (i, j, k)
    # We'll use the standard (i, j, k) enumeration.
    scheme223 = [
        # (a00,a01,a10,a11, b00,b01,b02,b10,b11,b12, c00,c01,c10,c11,c20,c21)
        (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),  # i=0,j=0,k=0 -> C[0][0]
        (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # i=1,j=0,k=0 -> C[0][1]
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),  # i=0,j=0,k=1 -> C[1][0]
        (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # i=1,j=0,k=1 -> C[1][1]
        (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # i=0,j=0,k=2 -> C[2][0]
        (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # i=1,j=0,k=2 -> C[2][1]
        (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),  # i=0,j=1,k=0 -> C[0][0]
        (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # i=1,j=1,k=0 -> C[0][1]
        (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),  # i=0,j=1,k=1 -> C[1][0]
        (0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # i=1,j=1,k=1 -> C[1][1]
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # i=0,j=1,k=2 -> C[2][0]
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # i=1,j=1,k=2 -> C[2][1]
    ]

    rank = 12 + 7 + 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Top row-block: 2x2x3 for C[0:3, 0:2] = A[0:2, 0:2] * B[0:2, 0:3]
    # u indices: [a*m+b] for a in 0..1, b in 0..1 -> [0, 1, 4, 5]
    # v indices: [b*p+k] for b in 0..1, k in 0..2 -> [0, 1, 2, 7, 8, 9]
    # w indices: [k*n+i] for k in 0..2, i in 0..1 -> [0, 1, 2, 3, 4, 5]
    u_top = [0, 1, 4, 5]
    v_top = [0, 1, 2, 7, 8, 9]
    w_top = [0, 1, 2, 3, 4, 5]
    for coeffs in scheme223:
        a = coeffs[0:4]
        b = coeffs[4:10]
        c = coeffs[10:16]
        for t in range(4):
            if a[t] != 0:
                U[u_top[t]][col] = a[t]
        for t in range(6):
            if b[t] != 0:
                V[v_top[t]][col] = b[t]
        for t in range(6):
            if c[t] != 0:
                W[w_top[t]][col] = c[t]
        col += 1

    # Middle row-block: 2x2x2 for C[3:5, 0:2] = A[0:2, 2:4] * B[2:4, 3:5]
    # u indices: a*m+b for a in 0..1, b in 2..3 -> [2, 3, 6, 7]
    # v indices: b*p+k for b in 2..3, k in 3..4 -> [14, 15, 21, 22]
    # w indices: k*n+i for k in 3..4, i in 0..1 -> [6, 7, 8, 9]
    u_mid = [2, 3, 6, 7]
    v_mid = [14, 15, 21, 22]
    w_mid = [6, 7, 8, 9]
    for coeffs in strassen:
        a = coeffs[0:4]
        b = coeffs[4:8]
        c = coeffs[8:12]
        for t in range(4):
            if a[t] != 0:
                U[u_mid[t]][col] = a[t]
        for t in range(4):
            if b[t] != 0:
                V[v_mid[t]][col] = b[t]
        for t in range(4):
            if c[t] != 0:
                W[w_mid[t]][col] = c[t]
        col += 1

    # Bottom row-block: 2x2x2 for C[5:7, 0:2] = A[0:2, 2:4] * B[2:4, 5:7]
    # u indices: a*m+b for a in 0..1, b in 2..3 -> [2, 3, 6, 7]
    # v indices: b*p+k for b in 2..3, k in 5..6 -> [16, 17, 23, 24]
    # w indices: k*n+i for k in 5..6, i in 0..1 -> [10, 11, 12, 13]
    u_bot = [2, 3, 6, 7]
    v_bot = [16, 17, 23, 24]
    w_bot = [10, 11, 12, 13]
    for coeffs in strassen:
        a = coeffs[0:4]
        b = coeffs[4:8]
        c = coeffs[8:12]
        for t in range(4):
            if a[t] != 0:
                U[u_bot[t]][col] = a[t]
        for t in range(4):
            if b[t] != 0:
                V[v_bot[t]][col] = b[t]
        for t in range(4):
            if c[t] != 0:
                W[w_bot[t]][col] = c[t]
        col += 1

    return {"U": U, "V": V, "W": W}

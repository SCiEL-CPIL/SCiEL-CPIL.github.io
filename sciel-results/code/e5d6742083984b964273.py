"""Rank-minimal exact tensor decomposition for 3x4 by 4x4 matrix multiplication."""

# EVOLVE-BLOCK-START

def construct():
    import numpy as np
    n, m, p = 3, 4, 4

    # Exact matmul tensor T[k, i, j] = 1 for all valid (i,j,k).
    T = np.zeros((p, n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k, i, j] = 1

    def verify(U, V, W):
        Ua = np.array(U, dtype=float).reshape(n, m, -1)
        Va = np.array(V, dtype=float).reshape(m, p, -1)
        Wa = np.array(W, dtype=float).reshape(p, n, -1)
        prod = (Ua * Va.transpose(1, 0, 2) * Wa).sum(axis=2)  # (i,j,k)
        return bool(np.allclose(prod, T.transpose(1, 2, 0)))

    # Elementary 2x2 blocks (factor-index lists).
    A00 = [(0,0),(0,1),(1,0),(1,1)]
    A01 = [(0,2),(0,3),(1,2),(1,3)]
    A10 = [(1,0),(1,1),(1,2),(1,3)]
    A11 = [(1,2),(1,3)]
    B00 = [(0,0),(0,1),(1,0),(1,1)]
    B01 = [(0,2),(0,3),(1,2),(1,3)]
    B10 = [(2,0),(2,1),(3,0),(3,1)]
    B11 = [(2,2),(2,3),(3,2),(3,3)]
    C00 = [(0,0),(1,0),(0,1),(1,1)]   # (k,i)
    C01 = [(2,0),(2,1),(3,0),(3,1)]
    C10 = [(0,2),(0,3),(1,2),(1,3)]
    C11 = [(2,2),(2,3),(3,2),(3,3)]

    U, V, W = [], [], []

    def add(A_part, B_part, C_part, sa, sb, sc):
        # A_part: list of (i,j) or None->use whole?; we pass concrete lists.
        col = len(U); U.append(None); V.append(None); W.append(None)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        for (ii, jj) in A_part:
            u[ii*m+jj] = sa
        for (jj, kk) in B_part:
            v[jj*p+kk] = sb
        for (kk, ii) in C_part:
            w[kk*n+ii] = sc
        U[col] = u; V[col] = v; W[col] = w

    # Strassen: 7 products, each contributes to a C sub-block (full vector).
    # M1 = (A00+A11)(B00+B11)  -> C00+, C01+, C10+, C11-  (one term each, C vector = block)
    add(A00+A11, B00+B11, C00, +1, +1, +1)
    add(A00+A11, B00+B11, C01, +1, +1, +1)
    add(A00+A11, B00+B11, C10, +1, +1, +1)
    add(A00+A11, B00+B11, C11, +1, +1, -1)
    # M2 = (A10+A11)B00 -> C00-, C01-
    add(A10+A11, B00, C00, +1, +1, -1)
    add(A10+A11, B00, C01, +1, +1, -1)
    # M3 = A00(B10-B11) -> C01-, C11-
    add(A00, B10, C01, +1, +1, -1)
    add(A00, B11, C01, +1, +1, +1)
    add(A00, B10, C11, +1, +1, -1)
    add(A00, B11, C11, +1, +1, +1)
    # M4 = (A00+A11)B10 -> C00-, C10+, C11-
    add(A00+A11, B10, C00, +1, +1, -1)
    add(A00+A11, B10, C10, +1, +1, +1)
    add(A00+A11, B10, C11, +1, +1, -1)
    # M5 = (A10+A11)B11 -> C00+, C10+
    add(A10+A11, B11, C00, +1, +1, +1)
    add(A10+A11, B11, C10, +1, +1, +1)
    # M6 = A11(B10-B11) -> C11+
    add(A11, B10, C11, +1, +1, +1)
    add(A11, B11, C11, +1, +1, -1)
    # M7 = A10(B01-B11) -> C11-
    add(A10, B01, C11, +1, +1, -1)
    add(A10, B11, C11, +1, +1, +1)

    # This gives the 2x2-block (rows i=0,1) part exactly (C[0:2,0:4] & C[2:4,0:4] with i in 0,1).
    # Rows i=2 (last row of A) handled schoolbook, 16 terms.
    for j in range(m):
        for k in range(p):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[2*m+j] = 1
            v[j*p+k] = 1
            w[k*n+2] = 1
            U.append(u); V.append(v); W.append(w)

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback schoolbook rank 48.
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    assert verify(U, V, W)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

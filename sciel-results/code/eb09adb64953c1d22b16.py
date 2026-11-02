import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # Build rank-48 solution:
    # 2 j-blocks × (3 Strassen-7 + 1 improved-2×2×1 rank-3) = 2×24 = 48
    rank = 48
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00+=M1, C11+=M1
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (c+d)e -> C10+=M2, C11-=M2
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3 = a(f-h) -> C01+=M3, C11+=M3
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4 = d(g-e) -> C00+=M4, C10+=M4
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5 = (a+b)h -> C00-=M5, C01+=M5
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6 = (c-a)(e+f) -> C11+=M6
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7 = (b-d)(g+h) -> C00+=M7
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_improved_221(i0, j0, k0):
        """2x2 * 2x1 with rank 3 (not 4).
        t0 = a00*b0 + a10*b1
        t1 = a01*b0 + a11*b1
        t2 = (a00-a01)*(b0-b1)
        C0 = t0 + t2, C1 = t1 + t2
        """
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        # t0 = a00*b0 + a10*b1
        U[a, col] += 1; U[c, col] += 1
        V[e, col] += 1; V[g, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # t1 = a01*b0 + a11*b1
        U[b, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[g, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # t2 = (a00-a01)*(b0-b1)
        U[a, col] += 1; U[b, col] -= 1
        V[e, col] += 1; V[g, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1

    # Build the rank-48 solution
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_improved_221(0, j_start, 6)

    if col != rank:
        raise RuntimeError(f"Expected col={rank}, got {col}")

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback: rank-50 with schoolbook 2×2×1
    rank50 = 50
    U50 = np.zeros((n*m, rank50), dtype=np.int32)
    V50 = np.zeros((m*p, rank50), dtype=np.int32)
    W50 = np.zeros((p*n, rank50), dtype=np.int32)
    col50 = 0

    def add_strassen_222_50(i0, j0, k0):
        nonlocal col50
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        U50[a, col50] += 1; U50[d, col50] += 1
        V50[e, col50] += 1; V50[h, col50] += 1
        W50[c00, col50] += 1; W50[c11, col50] += 1
        col50 += 1
        U50[c, col50] += 1; U50[d, col50] += 1
        V50[e, col50] += 1
        W50[c10, col50] += 1; W50[c11, col50] -= 1
        col50 += 1
        U50[a, col50] += 1
        V50[f, col50] += 1; V50[h, col50] -= 1
        W50[c01, col50] += 1; W50[c11, col50] += 1
        col50 += 1
        U50[d, col50] += 1
        V50[g, col50] += 1; V50[e, col50] -= 1
        W50[c00, col50] += 1; W50[c10, col50] += 1
        col50 += 1
        U50[a, col50] += 1; U50[b, col50] += 1
        V50[h, col50] += 1
        W50[c00, col50] -= 1; W50[c01, col50] += 1
        col50 += 1
        U50[c, col50] += 1; U50[a, col50] -= 1
        V50[e, col50] += 1; V50[f, col50] += 1
        W50[c11, col50] += 1
        col50 += 1
        U50[b, col50] += 1; U50[d, col50] -= 1
        V50[g, col50] += 1; V50[h, col50] += 1
        W50[c00, col50] += 1
        col50 += 1

    def add_schoolbook_221_50(i0, j0, k0):
        nonlocal col50
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        U50[a, col50] += 1; V50[e, col50] += 1; W50[c00, col50] += 1; col50 += 1
        U50[b, col50] += 1; V50[g, col50] += 1; W50[c00, col50] += 1; col50 += 1
        U50[c, col50] += 1; V50[e, col50] += 1; W50[c10, col50] += 1; col50 += 1
        U50[d, col50] += 1; V50[g, col50] += 1; W50[c10, col50] += 1; col50 += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222_50(0, j_start, k_start)
        add_schoolbook_221_50(0, j_start, 6)

    if verify(U50, V50, W50):
        return {
            "U": U50.tolist(),
            "V": V50.tolist(),
            "W": W50.tolist()
        }

    # Ultimate fallback: full schoolbook rank 56
    rank56 = n * m * p
    U56 = np.zeros((n*m, rank56), dtype=np.int32)
    V56 = np.zeros((m*p, rank56), dtype=np.int32)
    W56 = np.zeros((p*n, rank56), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U56[i*m + j, idx] = 1
                V56[j*p + k, idx] = 1
                W56[k*n + i, idx] = 1
                idx += 1
    return {
        "U": U56.tolist(),
        "V": V56.tolist(),
        "W": W56.tolist()
    }

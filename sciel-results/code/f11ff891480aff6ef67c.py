import numpy as np
import time

DIMENSIONS = [2, 3, 6]


def _build():
    n, m, p = DIMENSIONS  # 2, 3, 6
    num_A = n * m        # 6
    num_B = m * p        # 18
    num_C = p * n        # 12

    rank = 27  # 3 Strassen blocks (21) + 3 outer-product blocks (6)

    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))

    def idx_a(i, j):
        return i * m + j

    def idx_b(j, k):
        return j * p + k

    def idx_c(k, i):
        return k * n + i

    # Strassen 2x2x2 core (7 multiplications), offset k0.
    # A 2x2: A00=A[0,k0], A01=A[0,k0+1], A10=A[1,k0], A11=A[1,k0+1]
    # B 2x2: B00=B[k0,0], B01=B[k0,1], B10=B[k0+1,0], B11=B[k0+1,1]
    # C 2x2: C00=C[0,k0], C01=C[0,k0+1], C10=C[1,k0], C11=C[1,k0+1]
    def strassen(k0, c0):
        a00, a01 = idx_a(0, k0), idx_a(0, k0 + 1)
        a10, a11 = idx_a(1, k0), idx_a(1, k0 + 1)
        b00, b01 = idx_b(k0, 0), idx_b(k0, 1)
        b10, b11 = idx_b(k0 + 1, 0), idx_b(k0 + 1, 1)
        c00, c01 = idx_c(0, k0), idx_c(0, k0 + 1)
        c10, c11 = idx_c(1, k0), idx_c(1, k0 + 1)

        M1 = [(a10 + a11, b00, 1), (a10 + a11, b01, 1)]
        M2 = [(a00 + a01, b00, 1), (a00 + a01, b01, 1)]
        M3 = [(a11, b00 - b10, 1), (a11, b01 - b11, 1)]
        M4 = [(a00, b10 + b11, 1), (a00, b10 + b11, 1)]
        M5 = [(a00 + a11, b00, 1), (a00 + a11, b11, 1)]
        M6 = [(a01 - a11, b10 + b11, 1), (a01 - a11, b10 + b11, 1)]
        M7 = [(a10 - a00, b00 + b01, 1), (a10 - a00, b00 + b01, 1)]

        Cc = {
            c00: [(1, M1), (1, M4), (1, M5), (-1, M6)],
            c01: [(1, M5), (1, M7)],
            c10: [(1, M2), (1, M5), (-1, M3)],
            c11: [(1, M1), (-1, M3), (1, M4), (1, M7)],
        }

        for c in (c00, c01, c10, c11):
            for (sign, M) in Cc[c]:
                for (a, b, w) in M:
                    U[a, c0] += sign * w
                    V[b, c0] += sign * w
                    W[c, c0] += 1
                    c0 += 1
        return c0

    # Outer product 2x1x2 core (2 multiplications), offset k0.
    # A: A0=A[0,k0], A1=A[1,k0]; B: B0=B[k0,0], B1=B[k0,1]
    # C: C0=C[0,k0], C1=C[0,k0+1], C2=C[1,k0], C3=C[1,k0+1]
    def outer(k0, c0):
        a0, a1 = idx_a(0, k0), idx_a(1, k0)
        b0, b1 = idx_b(k0, 0), idx_b(k0, 1)
        c00, c01 = idx_c(0, k0), idx_c(0, k0 + 1)
        c10, c11 = idx_c(1, k0), idx_c(1, k0 + 1)
        M1 = [(a0, b0, 1), (a0, b1, 1)]
        M2 = [(a1, b0, 1), (a1, b1, 1)]
        for (a, b, w) in M1:
            U[a, c0] += w
            V[b, c0] += w
            W[c00, c0] += 1
            W[c10, c0] += 1
            c0 += 1
        for (a, b, w) in M2:
            U[a, c0] += w
            V[b, c0] += w
            W[c01, c0] += 1
            W[c11, c0] += 1
            c0 += 1
        return c0

    c0 = 0
    # j in {0,1}: 2x2x6 block = three 2x2x2 Strassen sub-blocks at k0 = 0,2,4
    for k0 in (0, 2, 4):
        c0 = strassen(k0, c0)
    # j = 2: 2x1x6 block = three 2x1x2 outer-product sub-blocks at k0 = 0,2,4
    for k0 in (0, 2, 4):
        c0 = outer(k0, c0)

    assert c0 == rank, (c0, rank)

    # Trim any fully-zero columns (defensive; should not trigger).
    keep = [c for c in range(rank)
            if U[:, c].any() or V[:, c].any() or W[:, c].any()]
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    return U, V, W


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    num_A, num_B, num_C = n * m, m * p, p * n
    Ua, Va, Wa = np.asarray(U, float), np.asarray(V, float), np.asarray(W, float)
    # Every valid triple must equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(np.dot(Ua[a], Va[b] * Wa[c]) - 1.0) > 1e-9:
                    return False
    # Every invalid (a,b,c) must equal 0 (full check: 6*18*12 = 1296).
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    for a in range(num_A):
        for b in range(num_B):
            for c in range(num_C):
                if (a, b, c) in valid:
                    continue
                if abs(np.dot(Ua[a], Va[b] * Wa[c])) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s wall-clock kill

    U, V, W = _build()

    # Confirm exactness; if (unexpectedly) invalid, fall back to schoolbook.
    if time.time() < deadline and _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36 (always valid).
    n, m, p = DIMENSIONS
    rank = n * m * p
    Us = np.zeros((n * m, rank))
    Vs = np.zeros((m * p, rank))
    Ws = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i * m + j, col] = 1
                Vs[j * p + k, col] = 1
                Ws[k * n + i, col] = 1
                col += 1
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

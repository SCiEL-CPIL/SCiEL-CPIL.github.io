import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0


def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 10.0)


def _strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0, 0] = 1; U[1, 0] = 1
    U[1, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[2, 4] = 1
    U[1, 5] = 1; U[2, 5] = -1
    U[3, 6] = 1; U[0, 6] = -1
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1; V[1, 1] = 1
    V[2, 2] = 1; V[1, 2] = -1
    V[0, 3] = 1; V[2, 3] = 1
    V[3, 4] = 1
    V[2, 5] = 1; V[3, 5] = 1
    V[1, 6] = 1; V[0, 6] = 1
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W


def _laderman_UV():
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
    U[1, 1] = 1; U[4, 1] = 1; V[1, 1] = 1; V[4, 1] = 1
    U[2, 2] = 1; U[5, 2] = 1; V[2, 2] = 1; V[5, 2] = 1
    U[3, 3] = 1; U[6, 3] = 1; V[3, 3] = 1; V[6, 3] = 1
    U[4, 4] = 1; U[7, 4] = 1; V[4, 4] = 1; V[7, 4] = 1
    U[5, 5] = 1; U[8, 5] = 1; V[5, 5] = 1; V[8, 5] = 1
    U[0, 6] = 1; U[1, 6] = 1; V[0, 6] = 1; V[1, 6] = 1
    U[3, 7] = 1; U[4, 7] = 1; V[3, 7] = 1; V[4, 7] = 1
    U[6, 8] = 1; U[7, 8] = 1; V[6, 8] = 1; V[7, 8] = 1
    U[1, 9] = 1; U[2, 9] = 1; V[1, 9] = 1; V[2, 9] = 1
    U[4, 10] = 1; U[5, 10] = 1; V[4, 10] = 1; V[5, 10] = 1
    U[7, 11] = 1; U[8, 11] = 1; V[7, 11] = 1; V[8, 11] = 1
    U[0, 12] = 1; U[2, 12] = 1; V[0, 12] = 1; V[2, 12] = 1
    U[3, 13] = 1; U[5, 13] = 1; V[3, 13] = 1; V[5, 13] = 1
    U[6, 14] = 1; U[8, 14] = 1; V[6, 14] = 1; V[8, 14] = 1
    U[0, 15] = 1; U[3, 15] = 1; U[6, 15] = 1
    V[0, 15] = 1; V[3, 15] = 1; V[6, 15] = 1
    U[1, 16] = 1; U[4, 16] = 1; U[7, 16] = 1
    V[1, 16] = 1; V[4, 16] = 1; V[7, 16] = 1
    U[2, 17] = 1; U[5, 17] = 1; U[8, 17] = 1
    V[2, 17] = 1; V[5, 17] = 1; V[8, 17] = 1
    U[0, 18] = 1; U[4, 18] = 1; U[8, 18] = 1
    V[0, 18] = 1; V[4, 18] = 1; V[8, 18] = 1
    U[3, 19] = 1; U[7, 19] = 1; U[8, 19] = 1
    V[3, 19] = 1; V[7, 19] = 1; V[8, 19] = 1
    U[1, 20] = 1; U[3, 20] = 1; U[5, 20] = 1
    V[1, 20] = 1; V[3, 20] = 1; V[5, 20] = 1
    U[2, 21] = 1; U[4, 21] = 1; U[6, 21] = 1
    V[2, 21] = 1; V[4, 21] = 1; V[6, 21] = 1
    for idx in range(9):
        U[idx, 22] = 1
        V[idx, 22] = 1
    return U, V


def _solve_W(U, V, n, m, p, r):
    W = np.zeros((p * n, r))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, r))
            for j in range(m):
                M[j, :] = U[i * m + j, :] * V[j * p + k, :]
            rhs = np.ones(m)
            sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            W[k * n + i, :] = sol
    return W


def _verify(U, V, W, n, m, p, iters=5):
    rng = np.random.default_rng(42)
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    for _ in range(iters):
        A = rng.integers(-3, 4, (n, m)).astype(float)
        B = rng.integers(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            scalar = np.dot(U[:, r], A.flatten()) * np.dot(V[:, r], B.flatten())
            if scalar != 0:
                C_computed += scalar * W[:, r].reshape(n, p)
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return False
    return True


def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    import math
    n1 = int(math.isqrt(Ua.shape[0])); m1 = int(math.isqrt(Va.shape[0])); p1 = int(math.isqrt(Wa.shape[0]))
    n2 = int(math.isqrt(Ub.shape[0])); m2 = int(math.isqrt(Vb.shape[0])); p2 = int(math.isqrt(Wb.shape[0]))
    r1 = Ua.shape[1]; r2 = Ub.shape[1]
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r = r1 * r2
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1 * m1 + j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2 * m2 + j2, r2i]
                            if b == 0: continue
                            i = i1 * n2 + i2; j = j1 * m2 + j2
                            U[i * m + j, r_idx] = a * b
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1 * p1 + k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2 * p2 + k2, r2i]
                            if b == 0: continue
                            j = j1 * m2 + j2; k = k1 * p2 + k2
                            V[j * p + k, r_idx] = a * b
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1 * n1 + i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2 * n2 + i2, r2i]
                            if b == 0: continue
                            k = k1 * p2 + k2; i = i1 * n2 + i2
                            W[k * n + i, r_idx] = a * b
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    def fallback():
        U, V, W = _schoolbook(6, 6, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if _check_time():
        return fallback()

    best = None
    try:
        U3, V3 = _laderman_UV()
        W3 = _solve_W(U3, V3, 3, 3, 3, 23)
        if _verify(U3, V3, W3, 3, 3, 3):
            U2, V2, W2 = _strassen_2x2()
            if _verify(U2, V2, W2, 2, 2, 2):
                U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
                if _verify(U, V, W, 6, 6, 6):
                    best = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    if best is not None and not _check_time():
        return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}

    try:
        U3s, V3s, W3s = _schoolbook(3, 3, 3)
        U2, V2, W2 = _strassen_2x2()
        U, V, W = _kron_decompose(U3s, V3s, W3s, U2, V2, W2)
        if _verify(U, V, W, 6, 6, 6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    return fallback()

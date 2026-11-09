import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _max_error(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    maxerr = 0.0
    for i in range(n):
        for j in range(m):
            prod = V3[j] * W3[:, i]
            P = U3[i, j] @ prod.T
            e = np.max(np.abs(P - 1.0))
            if e > maxerr:
                maxerr = e
    return maxerr


def _grad(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    E = np.empty((n, m, p))
    for i in range(n):
        for j in range(m):
            prod = V3[j] * W3[:, i]
            E[i, j] = U3[i, j] @ prod.T - 1.0
    Ug = np.empty_like(U)
    Vg = np.empty_like(V)
    Wg = np.empty_like(W)
    for i in range(n):
        for j in range(m):
            prod = V3[j] * W3[:, i]
            Ug[i * m + j] = 2.0 * (E[i, j] @ prod)
    for j in range(m):
        for k in range(p):
            A = U3[:, j] * W3[k]
            Vg[j * p + k] = 2.0 * (E[:, j, k] @ A)
    for i in range(n):
        for k in range(p):
            B = U3[i] * V3[:, k]
            Wg[k * n + i] = 2.0 * (E[i, :, k] @ B)
    return Ug, Vg, Wg


def _round_halfint(arr):
    rounded = np.round(arr * 2) / 2.0
    if np.max(np.abs(arr - rounded)) < 1e-6:
        return rounded, True
    return arr, False


def _search_rank(rank, n, m, p, seed, max_iter, deadline, lr0, scale0):
    np.random.seed(seed)
    r = rank
    U = np.random.randn(n * m, r) * scale0
    V = np.random.randn(m * p, r) * scale0
    W = np.random.randn(p * n, r) * scale0
    scale = (n * m * p / r) ** (1.0 / 3.0)
    U *= scale
    V *= scale
    W *= scale

    b1, b2, eps = 0.9, 0.999, 1e-8
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    t = 0
    lr = lr0
    best_err = np.inf

    for it in range(max_iter):
        if time.time() > deadline:
            break
        t += 1
        Ug, Vg, Wg = _grad(U, V, W, n, m, p)
        mU = b1 * mU + (1 - b1) * Ug; vU = b2 * vU + (1 - b2) * Ug * Ug
        mV = b1 * mV + (1 - b1) * Vg; vV = b2 * vV + (1 - b2) * Vg * Vg
        mW = b1 * mW + (1 - b1) * Wg; vW = b2 * vW + (1 - b2) * Wg * Wg
        mUh = mU / (1 - b1 ** t); vUh = vU / (1 - b2 ** t)
        mVh = mV / (1 - b1 ** t); vVh = vV / (1 - b2 ** t)
        mWh = mW / (1 - b1 ** t); vWh = vW / (1 - b2 ** t)
        U -= lr * mUh / (np.sqrt(vUh) + eps)
        V -= lr * mVh / (np.sqrt(vVh) + eps)
        W -= lr * mWh / (np.sqrt(vWh) + eps)
        # gentle lr decay
        lr = lr0 * (0.9995 ** it)

        if it % 150 == 0:
            err = _max_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
            if err < 1e-7:
                # polish pass (high precision)
                for _ in range(3000):
                    if time.time() > deadline:
                        break
                    Ug, Vg, Wg = _grad(U, V, W, n, m, p)
                    U -= 1e-4 * Ug
                    V -= 1e-4 * Vg
                    W -= 1e-4 * Wg
                err = _max_error(U, V, W, n, m, p)
                if err < 1e-9:
                    return U, V, W
    err = _max_error(U, V, W, n, m, p)
    if err < 1e-9:
        return U, V, W
    return None


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill
    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Guaranteed exact integer fallback (schoolbook)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Search for a valid half-integer decomposition with rank < base_rank.
    # Diverse initializations / learning rates to explore the non-convex landscape.
    lr_choices = [0.02, 0.05, 0.01]
    scale_choices = [0.3, 0.6, 1.0]

    for rank in range(base_rank - 1, max(base_rank - 24, 27), -1):
        if time.time() > deadline:
            break
        for seed in range(8):
            if time.time() > deadline:
                break
            lr0 = lr_choices[seed % len(lr_choices)]
            sc0 = scale_choices[(seed // len(lr_choices)) % len(scale_choices)]
            res = _search_rank(rank, n, m, p, seed, max_iter=1200,
                               deadline=deadline, lr0=lr0, scale0=sc0)
            if res is not None:
                U2, V2, W2 = res
                Ur, ok1 = _round_halfint(U2)
                Vr, ok2 = _round_halfint(V2)
                Wr, ok3 = _round_halfint(W2)
                if ok1 and ok2 and ok3:
                    if _max_error(Ur, Vr, Wr, n, m, p) < 1e-9:
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

    # Safe exact fallback
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

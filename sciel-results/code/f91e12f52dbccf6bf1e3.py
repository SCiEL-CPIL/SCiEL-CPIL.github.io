"""Matrix multiplication 2x3x4 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 140.0  # safety margin before 150s kill

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _reconstruct(U, V, W, n, m, p):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    U = np.reshape(U, (n, m, r))
    V = np.reshape(V, (m, p, r))
    W = np.reshape(W, (p, n, r))
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    T = _reconstruct(U, V, W, n, m, p)
    Tt = _target_tensor()
    return np.max(np.abs(T - Tt)) < tol

def _round_half(x):
    return np.round(2.0 * x) / 2.0

def _try_rank(r, n, m, p, time_budget, rng):
    """Try to find a rank-r decomposition. Returns (U,V,W) lists or None."""
    Tt = _target_tensor()
    U = rng.standard_normal((n * m, r))
    V = rng.standard_normal((m * p, r))
    W = rng.standard_normal((p * n, r))
    best_err = np.inf
    lr = 0.05
    t0 = time.time()
    # simple gradient-free / gradient descent on the product
    for it in range(4000):
        if time.time() - t0 > time_budget:
            break
        T = _reconstruct(U, V, W, n, m, p)
        err = np.mean((T - Tt) ** 2)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-12:
            break
        # gradients
        Ur = np.reshape(U, (n, m, r))
        Vr = np.reshape(V, (m, p, r))
        Wr = np.reshape(W, (p, n, r))
        dT = 2.0 * (T - Tt) / (n * m * p)
        dU = np.einsum('ijk,jkc,kic->ijc', dT, Vr, Wr)
        dV = np.einsum('ijk,ijc,kic->jkc', dT, Ur, Wr)
        dW = np.einsum('ijk,ijc,jkc->kic', dT, Ur, Vr)
        U -= lr * dU.reshape(n * m, r)
        V -= lr * dV.reshape(m * p, r)
        W -= lr * dW.reshape(p * n, r)
    U, V, W = best
    # round to half-integers
    Ur = _round_half(U); Vr = _round_half(V); Wr = _round_half(W)
    if _verify(Ur, Vr, Wr, n, m, p):
        return Ur, Vr, Wr
    return None

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    # fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    # try lower ranks
    for r in range(max(n * m, m * p, p * n), n * m * p):
        if time.time() - t_start > DEADLINE - 5:
            break
        budget = min(10.0, max(1.0, (DEADLINE - 5) - (time.time() - t_start)))
        found = None
        for attempt in range(3):
            if time.time() - t_start > DEADLINE - 5:
                break
            rng = np.random.default_rng(1000 * r + attempt)
            found = _try_rank(r, n, m, p, budget / 3.0, rng)
            if found is not None:
                break
        if found is not None:
            Ur, Vr, Wr = found
            U = Ur.tolist(); V = Vr.tolist(); W = Wr.tolist()
            return {"U": U, "V": V, "W": W}
    return best

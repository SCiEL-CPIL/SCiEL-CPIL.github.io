import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_TIME_LIMIT = 110.0   # safety margin before 120s kill


def _build_tensor(n, m, p):
    """Participation tensor T[a,b,c]=1 iff a%m == b//p.
    Shape (n*m, m*p, p*n); T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from(U, V, W):
    rank = U.shape[1]
    return (U.T @ V) * W.T  # (n*m, m*p, p*n)


def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _tensor_from(U, V, W)
    return np.max(np.abs(T - _build_tensor(n, m, p))) <= tol


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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _structured(n, m, p, mU, mV, mW, seed=0):
    """Each j-slice of U uses only the j-block of mU; each j-slice of V uses
    the j-block of mV; each j-slice of W uses the j-block of mW. This respects
    the tensor's block structure (the shared-index constraint)."""
    rank = mU + mV + mW
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    rng = np.random.default_rng(seed)
    # U block: rows i*m+j use cols [j*p .. j*p+mU)
    for j in range(m):
        for i in range(n):
            U[i * m + j, j * p:(j + 1) * p] = rng.standard_normal(mU)
    # V block: rows j*p+k use cols [mU + k*n .. mU + k*n + mV)
    for k in range(p):
        for j in range(m):
            V[j * p + k, mU + k * n:mU + k * n + mV] = rng.standard_normal(mV)
    # W block: rows k*n+i use cols [mU + mV + j*p .. mU + mV + j*p + mW)
    for j in range(m):
        for k in range(p):
            W[k * n + i if False else k * n + i, mU + mV + j * p:mU + mV + j * p + mW] = rng.standard_normal(mW)
    return U, V, W


def _polish(U, V, W, n, m, p, iters=200, lr=0.05):
    """Smooth NLLS polish: minimize ||T_est - T||^2 with coordinate descent.
    Uses a smooth surrogate (least squares) with adaptive step."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    def loss():
        return np.sum((_tensor_from(U, V, W) - T) ** 2)

    best = loss()
    for it in range(iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        # coordinate descent over U, V, W
        for _ in range(3):
            # U
            for a in range(nm):
                i, j = divmod(a, m)
                grad = np.zeros(rank)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    diff = np.sum(U[a] * V[b] * W[c]) - T[a, b, c]
                    grad += diff * V[b] * W[c]
                U[a] -= lr * grad
            # V
            for b in range(mp):
                j, k = divmod(b, p)
                grad = np.zeros(rank)
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    diff = np.sum(U[a] * V[b] * W[c]) - T[a, b, c]
                    grad += diff * U[a] * W[c]
                V[b] -= lr * grad
            # W
            for c in range(pn):
                k, i = divmod(c, n)
                grad = np.zeros(rank)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    diff = np.sum(U[a] * V[b] * W[c]) - T[a, b, c]
                    grad += diff * U[a] * V[b]
                W[c] -= lr * grad
        cur = loss()
        if cur < best:
            best = cur
        if best < 1e-10:
            break
    return U, V, W


def _try_rank(n, m, p, mU, mV, mW, seeds=(0, 1, 2)):
    rank = mU + mV + mW
    for seed in seeds:
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return None
        U, V, W = _structured(n, m, p, mU, mV, mW, seed)
        U, V, W = _polish(U, V, W, n, m, p, iters=300, lr=0.05)
        if _verify(U, V, W, n, m, p, tol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)

    # Try a ladder of structured ranks from low to high.
    # mU per j-block, mV per k-block, mW per j-block.
    candidates = [
        (1, 1, 1),   # rank 9
        (1, 2, 1),   # rank 12
        (2, 1, 1),   # rank 12
        (1, 1, 2),   # rank 12
        (2, 2, 1),   # rank 15
        (2, 1, 2),   # rank 15
        (1, 2, 2),   # rank 15
        (2, 2, 2),   # rank 18
        (3, 2, 2),   # rank 21
        (2, 3, 2),   # rank 21
        (2, 2, 3),   # rank 21
        (3, 3, 2),   # rank 24
        (3, 2, 3),   # rank 24
        (2, 3, 3),   # rank 24
        (3, 3, 3),   # rank 27
    ]
    for (mU, mV, mW) in candidates:
        if time.time() - _START_TIME > _TIME_LIMIT - 8:
            break
        res = _try_rank(n, m, p, mU, mV, mW)
        if res is not None:
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                best = res
                break  # found a valid lower-rank; stop (we iterate low->high)

    # Final deadline guard: ensure we return something valid.
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)
    return best

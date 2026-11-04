"""Search for a rank-minimal exact tensor decomposition of 3x3 x 3x4 matrix mult."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _slice_rank(T, mode):
    n0, n1, n2 = T.shape
    if mode == 0:
        mats = [T[a, :, :] for a in range(n0)]
    elif mode == 1:
        mats = [T[:, b, :] for b in range(n1)]
    else:
        mats = [T[:, :, c] for c in range(n2)]
    flat = np.array([m.flatten() for m in mats])
    return int(np.linalg.matrix_rank(flat, tol=1e-9))

def _verify_int(Ur, Vr, Wr, T):
    # Ur,Vr,Wr are half-integers; check exact decomposition via integer math.
    Ur2 = np.rint(Ur * 2).astype(np.int64)
    Vr2 = np.rint(Vr * 2).astype(np.int64)
    Wr2 = np.rint(Wr * 2).astype(np.int64)
    # R*8 = sum_c Ur2[a,c] Vr2[b,c] Wr2[c',c] must equal 8*T
    n0, n1, n2 = T.shape
    R8 = np.zeros((n0, n1, n2), dtype=np.int64)
    for c in range(Ur2.shape[1]):
        R8 += np.outer(Ur2[:, c], Vr2[:, c]) * Wr2[:, c][:, None]
    return np.array_equal(R8, (8 * T).astype(np.int64))

def _als_run(T, r, seed):
    rng = np.random.RandomState(seed)
    n0, n1, n2 = T.shape
    U = rng.randn(n0, r)
    V = rng.randn(n1, r)
    W = rng.randn(n2, r)
    # Precompute flattened target slices
    Tw = [T[:, :, c].flatten() for c in range(n2)]
    Tu = [T[a, :, :].flatten() for a in range(n0)]
    Tv = [T[:, b, :].flatten() for b in range(n1)]
    for it in range(400):
        # update W
        Mmat = np.empty((n0 * n1, r))
        for c in range(r):
            Mmat[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        for c in range(n2):
            W[c, :] = np.linalg.lstsq(Mmat, Tw[c], rcond=None)[0]
        # update U
        Mmat2 = np.empty((n1 * n2, r))
        for c in range(r):
            Mmat2[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        for a in range(n0):
            U[a, :] = np.linalg.lstsq(Mmat2, Tu[a], rcond=None)[0]
        # update V
        Mmat3 = np.empty((n0 * n2, r))
        for c in range(r):
            Mmat3[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        for b in range(n1):
            V[b, :] = np.linalg.lstsq(Mmat3, Tv[b], rcond=None)[0]
    # residual
    R = np.zeros((n0, n1, n2))
    for c in range(r):
        R += np.outer(U[:, c], V[:, c]) * W[:, c][:, None]
    err = np.max(np.abs(R - T))
    if err < 1e-6:
        Ur = np.round(U * 2) / 2.0
        Vr = np.round(V * 2) / 2.0
        Wr = np.round(W * 2) / 2.0
        if _verify_int(Ur, Vr, Wr, T):
            return Ur, Vr, Wr
    return None

def construct():
    T = _build_target()
    n, m, p = DIMENSIONS
    lb = max(_slice_rank(T, 0), _slice_rank(T, 1), _slice_rank(T, 2))

    deadline = time.time() + 100
    best = None
    seed = 0
    for r in range(lb, lb + 15):
        if time.time() > deadline:
            break
        attempts = 250 if r <= lb + 2 else 120
        for _ in range(attempts):
            if time.time() > deadline:
                break
            res = _als_run(T, r, seed)
            seed += 1
            if res is not None:
                best = (r, res)
                break
        if best is not None:
            break

    if best is None:
        # fallback: schoolbook (valid integer decomposition)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        r, (Ur, Vr, Wr) = best
        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
